#!/usr/bin/env python3
"""
Github Actions Workflow wrapper for Ruff.

While Ruff does have an output format for Github Actions Workflow syntax, it is lacking in a few different ways:
- No means of filtering results to changes.
- File paths are always full absolute paths.
- Does not present fix suggestions.

This script aims to make up for these shortcomings and add a few other niceties, such as text coloring.

In order to get the most usable information from ruff, ruff is run with the output mode set to sarif.
Sarif specification: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
"""

import argparse
import copy
import difflib
import functools
import io
import itertools
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import (
	unquote as urlunquote,
	urlparse,
)

try:
	from termcolor import colored as termcolor_str
except ImportError:
	def termcolor_str(text, *args, **kwargs):  # noqa: ARG001
		"""
		Passthrough text if termcolor is not available.

		Args:
			text: String to (not) format.
			*args: Ignored.
			**kwargs: Ignored.

		Returns:
			text unchanged.
		"""
		return text


# For parsing diffs from git
RE_DIFF_FILENAME = re.compile(r"^\+\+\+\ [^/]+/(.*)")
RE_DIFF_HUNK_LINES = re.compile(r"^@@ -[0-9,]+ \+(\d+)(,(\d+))?")

# This environment variable is set if we're running in a Github Action
RUNNING_GHA = os.environ.get("GITHUB_ACTION")

if RUNNING_GHA:
	def gha_color(text, *args, **kwargs):  # noqa: ARG001
		"""
		Passthrough text if running in GHA.

		Args:
			text: String to (not) format.
			*args: Ignored.
			**kwargs: Ignored.

		Returns:
			text unchanged.
		"""
		return text
else:
	gha_color = termcolor_str

GHA_LEVELS = {
	"error":   gha_color("::error",   color="red",        attrs=["bold"]),
	"warning": gha_color("::warning", color="yellow",     attrs=["bold"]),
	"notice":  gha_color("::notice",                      attrs=["bold"]),
}
GHA_DEBUG =    gha_color("::debug",   color="blue",       attrs=["bold"])


@functools.total_ordering
class LineRange:
	"""Range of lines in a file with optional column tracking."""

	def __init__(self, start, end=None, count=None, start_col=None, end_col=None):
		"""
		Initialize a LineRange.

		Args:
			start: Start line of line range.
			end: End line of line range. Either end or count must be specified, but not both.
			count: Number of lines in the range. Either end or count must be specified, but not both.
			start_col: Column in start line where range starts. Optional.
			end_col: Column in end line where range ends. Optional.

		Raises:
			RuntimeError: If neither or both of end and count are specified.
		"""
		self._start = start
		self._end = end
		self._start_col = start_col
		self._end_col = end_col
		self._count = count

		if end is None:
			if count is None:
				raise RuntimeError('LineRange constructor called without end or count')
			self._end = start + count - 1
		elif count is None:
			self._count = end + 1 - start
		else:
			raise RuntimeError('LineRange constructor called with both end and count')

	@classmethod
	def from_sarif(cls, sarif_region):
		"""
		Create LineRange from a sarif region.

		Args:
			sarif_region: Parsed json of a sarif region.

		Returns:
			LineRange constructed from data in sarif_region.
		"""
		if not sarif_region:
			# no line number info
			return None

		start_line = sarif_region.get("startLine", None)
		end_line = sarif_region.get("endLine", None)
		if not start_line:
			# no start line
			return None

		start_column = sarif_region.get("startColumn", None)
		end_column = sarif_region.get("endColumn", None)

		return cls(start_line, end=end_line, start_col=start_column, end_col=end_column)

	@property
	def start(self):
		"""Start line of line range."""
		return self._start

	@property
	def count(self):
		"""Number of lines in line range."""
		return self._count

	@property
	def end(self):
		"""Last line of line range."""
		return self._end

	@property
	def start_col(self):
		"""Column in start line where range starts."""
		return self._start_col

	@property
	def end_col(self):
		"""Column in end line where range ends."""
		return self._end_col

	def __hash__(self):
		"""
		Generate hash value to aid in object comparison.

		Returns:
			Integer hash value.
		"""
		return hash((self.start, self.start_col, self.count, self.end_col))

	def __eq__(self, other):
		"""
		Determine object equality.

		Args:
			other: Object to compare to.

		Returns:
			True if objects are equal, False otherwise.
		"""
		if not isinstance(other, LineRange):
			return False
		return (
			(self.start == other.start)
			and (self.count == other.count)
			and (self.start_col == other.start_col)
			and (self.end_col == other.end_col)
		)

	def __lt__(self, other):
		"""
		Determine whether this object should be sorted before other object.

		Args:
			other: Object to compare to.

		Returns:
			True if this object sorts before other, False otherwise.
		"""
		if self.start < other.start:
			return True
		if self.start > other.start:
			return False

		if self.start_col is None and other.start_col is not None:
			# no col is always before 0 col
			return True
		if self.start_col is not None and other.start_col is None:
			# no col is always before 0 col
			return False
		if self.start_col is not None and other.start_col is not None:
			if self.start_col < other.start_col:
				return True
			if self.start_col > other.start_col:
				return False

		# smaller hunks before larger ones
		if self.count < other.count:
			return True
		if self.count > other.count:
			return False

		if self.end_col is None and other.end_col is not None:
			# no col is always before 0 col
			return True
		if self.end_col is not None and other.end_col is None:
			# no col is always before 0 col
			return False
		if self.end_col is not None and other.end_col is not None:
			# smaller hunks before larger ones
			if self.end_col < other.end_col:
				return True
			if self.end_col > other.end_col:
				return False

		# if we get all the way here, we're equal
		return False

	def intersects(self, other):
		"""
		Check for intersection with other LineRange, ignoring columns.

		Args:
			other: Object to compare to.

		Returns:
			True if LineRange objects intersect, False otherwise.
		"""
		# Method is written this way for readability
		if self.start > other.end:
			# we start after other ends
			return False
		if self.end < other.start:  # noqa: SIM103
			# we end before other starts
			return False
		return True

	def overlaps(self, other):
		"""
		Check for intersection with other LineRange, accounting for columns.

		Args:
			other: Object to compare to.

		Returns:
			True if LineRange objects overlap, False otherwise.
		"""
		if self.intersects(other):
			if (self.start == other.end) and (self.start_col >= other.end_col):
				# we start on same line as other ends, but after
				return False
			if (self.end == other.start) and (self.end_col <= other.start_col):  # noqa: SIM103
				# we end on same line as other starts, but before
				return False
			return True
		return False


def convert_stdout(bytes_in):
	"""
	Ensure line from stdout is usable.

	Args:
		bytes_in: stdout line.

	Returns:
		String with proper encoding.
	"""
	try:
		return bytes_in.decode("utf-8").encode("utf-8")
	except AttributeError:
		return str(bytes_in)
	except UnicodeError:
		return str(bytes_in)


def git_diff(since_commit=None):
	"""
	Use git to generate a diff, then parse that diff to extract change locations.

	Args:
		since_commit: When specified, generated diff includes changes from commits since this commit all the way to
			HEAD. Otherwise, generated diff includes changes from HEAD to the current working tree.

	Returns:
		Ordered dictionary of lists of LineRange objects, keyed by file as pathlib Path objects.
	"""
	git_diff_args = ["git"]

	if since_commit is None:
		# TODO(#30): Investigate using diff-index with --cached instead of diff
		git_diff_args += [
			"diff",
			"HEAD",
		]
	else:
		git_diff_args += [
			"diff-tree",
			since_commit,
			"HEAD",
			"-r",
		]

	git_diff_args += [
		# full paths, not just filenames
		"--full-index",

		# track renames and copies
		"--find-renames",
		"--find-copies",
		"--find-copies-harder",

		# unified diff with no context
		"--patch",
		"--unified=0",

		# no external diff program
		"--no-ext-diff",

		# no colored output
		"--no-color",
	]

	env = os.environ.copy()
	# No pager, we need output directly from git.
	env["GIT_PAGER"] = ""
	# No locale shenanigans, please.
	env["LANGUAGE"] = "C"
	env["LC_ALL"] = "C"

	p = subprocess.Popen(git_diff_args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8")
	p.stdin.close()

	file = None
	changes = OrderedDict()
	for line in p.stdout:
		line = convert_stdout(line)  # noqa: PLW2901
		match = RE_DIFF_FILENAME.search(line)
		if match:
			file = match.group(1).rstrip("\r\n\t")
		if not file:
			# If we haven't found a file yet, no need to search for line nubmers
			continue
		file = Path(file)
		match = RE_DIFF_HUNK_LINES.search(line)
		if match:
			start_line = int(match.group(1))
			line_count = 1
			if match.group(3):
				line_count = int(match.group(3))
			if line_count == 0:
				continue
			if start_line == 0:
				continue
			changes.setdefault(file, []).append(LineRange(start_line, count=line_count))

	return changes


@contextmanager
def emit_gha_group(title):
	"""
	Context manager function to emit GHA workflow output group lines to stdout.

	Args:
		title: Group title.
	"""
	if RUNNING_GHA:
		print(f"::group::{title}")
	try:
		yield None
	finally:
		if RUNNING_GHA:
			print("::endgroup::")


def emit_gha_debug_message(message):
	"""
	Emit GHA workflow debug message to stdout.

	Args:
		message: Message to emit.
	"""
	print(f"{GHA_DEBUG}::{message}")


def emit_gha_message(level, properties, message):
	"""
	Emit GHA workflow message to stdout.

	Args:
		level: Level of message.
		properties: Properties of message (title, file, line, col, endLine, endColumn)
		message: Message to emit.
	"""
	level = GHA_LEVELS[level]
	props = ",".join([f"{param}={value}" for param, value in properties.items()])
	print("{} {}::{}".format(level, props, gha_color(message, attrs=["bold"])))


@functools.cache
def uri_to_path(uri):
	"""
	Convert file:// URI to pathlib Path.

	Args:
		uri: Path as file:// URI.

	Returns:
		pathlib Path relative to working directory.
	"""
	return Path(urlunquote(urlparse(uri).path)).relative_to(Path.cwd())


def get_check_files(git_changes):
	"""
	Get changed files that Ruff would operate on.

	Args:
		git_changes: list of (or dict keyed by) changed files as pathlib Paths

	Yields:
		Subset of git_changes on which ruff would operate.
	"""
	ruff_args = [
		"ruff",
		"check",
		"--show-files",
	]

	env = os.environ.copy()
	# No locale shenanigans, please.
	env["LANGUAGE"] = "C"
	env["LC_ALL"] = "C"

	p = subprocess.Popen(ruff_args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8")
	p.stdin.close()

	cwd = Path.cwd()
	for line in p.stdout:
		line = convert_stdout(line).rstrip("\r\n")  # noqa: PLW2901
		file = Path(line).relative_to(cwd)
		if file in git_changes:
			yield file


def invoke_check(file):
	"""
	Invoke ruff check.

	Args:
		file: File on which to operate.

	Returns:
		Sarif output from Ruff, as parsed json.
	"""
	ruff_args = [
		"ruff",
		"check",
		"--no-fix",
		"--no-fix-only",
		"--output-format", "sarif",
		str(file),
	]

	env = os.environ.copy()
	# No locale shenanigans, please.
	env["LANGUAGE"] = "C"
	env["LC_ALL"] = "C"

	p = subprocess.Popen(ruff_args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8")
	p.stdin.close()
	return json.load(p.stdout)


def invoke_format(file, region):
	"""
	Invoke ruff format.

	Args:
		file: File on which to operate.
		region: Line range on which to operate.

	Returns:
		Sarif output from Ruff, as parsed json.
	"""
	ruff_args = [
		"ruff",
		"format",
		"--check",
		"--preview",
		"--output-format", "sarif",
		"--range", f"{region.start}-{region.end}",
		str(file),
	]

	env = os.environ.copy()
	# No locale shenanigans, please.
	env["LANGUAGE"] = "C"
	env["LC_ALL"] = "C"

	p = subprocess.Popen(ruff_args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8")
	p.stdin.close()
	return json.load(p.stdout)


class RuffSuggestion:
	"""Parsed sarif fix from Ruff."""

	class Replacement:
		"""Parsed sarif replacement from Ruff."""

		def __init__(self, sarif_replacement):
			"""
			Initialize Replacement from sarif replacement.

			Args:
				sarif_replacement: Parsed json of a sarif replacement.
			"""
			self._region = LineRange.from_sarif(sarif_replacement.get("deletedRegion", {}))
			if not self._region.start_col:
				# replacement is unusable if no column info
				self._region = None
				self._new_content = None
			else:
				self._new_content = sarif_replacement.get("insertedContent", {}).get("text", "")

		@property
		def region(self):
			"""LineRange to be replaced. None if artifact change was unusable."""
			return self._region

		@property
		def new_content(self):
			"""Content to repalce region with. None if artifact change was unusable."""
			return self._new_content

	def __init__(self, sarif_fix):
		"""
		Initialize RuffSuggestion from sarif fix.

		Args:
			sarif_fix: Parsed json of a sarif fix.
		"""
		self._description = sarif_fix.get("description", {}).get("text", None)
		if not self._description:
			self._description = ""
		else:
			self._description = self._description

		self._changes = OrderedDict()

		for sarif_change in sarif_fix.get("artifactChanges", []):
			file = sarif_change.get("artifactLocation", {}).get("uri", None)
			if not file:
				continue
			file = uri_to_path(file)
			replacements = [RuffSuggestion.Replacement(r) for r in sarif_change.get("replacements", [])]
			replacements = [r for r in replacements if r.region]
			if not replacements:
				continue
			self._changes.setdefault(file, []).append(replacements)

	@property
	def description(self):
		"""Description of proposed fix."""
		return self._description

	@property
	def changes(self):
		"""
		Proposed changes as a mapping of list of lists of Replacement objects keyed to pathlib Path objects.

		Values are not flat lists, as each sarif fix can contain more than one set of replacements for
		an individual file.
		"""
		return self._changes

	def _apply_change(self, lines_old, lines_new, change):
		new_content = (
			lines_old[change.region.start - 1][: change.region.start_col - 1]
			+ change.new_content
			+ lines_old[change.region.end - 1][change.region.end_col - 1 :]
		)
		new_content = new_content.splitlines(keepends=True)

		lines_new[change.region.start - 1 : change.region.end] = new_content

	def _fmt_hunk_line_numbers(self, start, end):
		range_len = end - start
		start += 1
		if range_len == 1:
			return f"{start}"
		if not range_len:
			start -= 1
		return f"{start},{range_len}"

	def _get_trailing_space(self, line):
		space = ""
		# newline must be stripped so we don't format it
		if line[-1] == "\n":
			line = line[:-1]
		# split out trailing space so that it can be colored differently
		while line and line[-1] in {" ", "\t"}:
			space = line[-1] + space
			line = line[:-1]
		return (line, space)

	def _write_diff(self, sb, lines_old, lines_new, filepath, context_amt=3):
		# We don't generate the diff directly with difflib, as we want to color the text.
		header_written = False

		for hunk in difflib.SequenceMatcher(a=lines_old, b=lines_new).get_grouped_opcodes(context_amt):

			if not header_written:
				# write file header
				sb.write(termcolor_str(f"--- {filepath}", attrs=["bold"]))
				sb.write("\n")
				sb.write(termcolor_str(f"+++ {filepath}", attrs=["bold"]))
				sb.write("\n")
				header_written = True

			# write chunk header
			hunk_range_old = self._fmt_hunk_line_numbers(hunk[0][1], hunk[-1][2])
			hunk_range_new = self._fmt_hunk_line_numbers(hunk[0][3], hunk[-1][4])
			sb.write(termcolor_str(f"@@ -{hunk_range_old} +{hunk_range_new} @@", color="cyan"))
			sb.write("\n")

			for op, lines_old_start, lines_old_end, lines_new_start, lines_new_end in hunk:
				if op == "equal":
					for line in lines_old[lines_old_start:lines_old_end]:
						sb.write(" ")
						sb.write(line)
					continue
				if op in {"replace", "delete"}:
					for line in lines_old[lines_old_start:lines_old_end]:
						line, space = self._get_trailing_space(line)  # noqa: PLW2901
						sb.write(termcolor_str("-" + line, color="red"))
						if space:
							sb.write(termcolor_str(space, on_color="on_red"))
						sb.write("\n")
				if op in {"replace", "insert"}:
					for line in lines_new[lines_new_start:lines_new_end]:
						line, space = self._get_trailing_space(line)  # noqa: PLW2901
						sb.write(termcolor_str("+" + line, color="green"))
						if space:
							sb.write(termcolor_str(space, on_color="on_green"))
						sb.write("\n")

	def generate_diff(self):
		"""
		Generate diff of suggested changes.

		Returns:
			String containing diff.
		"""
		if not self.changes:
			return ""

		# Flatten change lists
		changes = OrderedDict()
		for path, changelists in self.changes.items():
			for changelist in changelists:
				if changelist:
					changes.setdefault(path, []).extend(changelist)

		# Sort change lists and detect overlapping changes
		changes_overlap = False
		def get_region(replacement):
			return replacement.region
		for changelist in changes.values():
			changelist.sort(key=get_region)
			if not changes_overlap:
				for c1, c2 in itertools.combinations(changelist, 2):
					if c1.region.overlaps(c2.region):
						changes_overlap = True
						emit_gha_debug_message("Overlapping changes, will split diffs")
						break

		sb = io.StringIO()

		sb.write(termcolor_str("Suggestion: ", color="light_blue", attrs=["bold"]))
		if self.description:
			sb.write(" ")
			sb.write(termcolor_str(self.description, attrs=["bold"]))
		sb.write("\n")

		lines_old = []
		with path.open(mode="rt", encoding="utf-8") as file:
			lines_old = file.readlines()

		if changes_overlap:
			# Split diffs if changes overlap.
			for change in changelist:
				lines_new = copy.copy(lines_old)
				self._apply_change(lines_old, lines_new, change)
				self._write_diff(sb, lines_old, lines_new, str(path))
		else:
			lines_new = copy.copy(lines_old)
			for change in reversed(changelist):
				self._apply_change(lines_old, lines_new, change)
			self._write_diff(sb, lines_old, lines_new, str(path))

		return sb.getvalue()


class RuffResult:
	"""Parsed sarif run result from Ruff."""

	def __init__(self, sarif_result, rule_dict=None, all_git_changes=None):
		"""
		Initialize RuffResult from sarif result.

		Args:
			sarif_result: Parsed json of a sarif result.
			rule_dict: Mapping of rule IDs to sarif rules as parsed json. Will enhance output if specified.
			all_git_changes: Mapping of pathlib Path objects to LineRange objects as returned by git_diff. Optional.
		"""
		self._title = None
		self._level = None
		self._summary = None
		self._suggestions = None

		self._locations = OrderedDict()

		for location in sarif_result.get("locations", []):
			location = location.get("physicalLocation", None)  # noqa: PLW2901
			if not location:
				continue

			result_location_file = location.get("artifactLocation", {}).get("uri", None)
			if not result_location_file:
				continue

			result_location_file = uri_to_path(result_location_file)

			sarif_region = location.get("region", None)
			result_location_region = LineRange.from_sarif(sarif_region)

			regions = self._locations.setdefault(result_location_file, [])
			if result_location_region not in regions:
				regions.append(result_location_region)
				regions.sort()

		if not self._locations:
			# no parsable locations, result unusable
			self._locations = None
			return

		is_change_relevant = all_git_changes is None
		if not is_change_relevant:
			for result_location_file, result_location_regions in self._locations.items():
				git_changes = all_git_changes.get(result_location_file, [])
				if not git_changes:
					continue
				if None in result_location_regions:
					# We have a result location with no/insufficient line info
					# Assume result applies to the entire file
					is_change_relevant = True
					break
				for result_location_region in result_location_regions:
					for git_change_region in git_changes:
						if result_location_region.intersects(git_change_region):
							is_change_relevant = True
							break
					if is_change_relevant:
						break
				if is_change_relevant:
					break

		if not is_change_relevant:
			self._locations = None
			return

		# Determine GHA level (error, warning, notice, debug)
		self._level = sarif_result.get("level", "error")
		sarif_kind = sarif_result.get("kind", None)
		if self._level in {"error", "warning"}:
			pass
		elif self._level == "note":
			self._level = "notice"
		elif sarif_kind == "review":
			self._level = "warning"
		elif sarif_kind in {"open", "informational"}:
			self._level = "debug"
		else:
			self._level = "error"

		self._summary = sarif_result.get("message", {}).get("text", None)
		if not self._summary:
			self._summary = "Check failure"

		result_rule_id = sarif_result.get("ruleId", None)

		if result_rule_id == "unformatted":
			self._title = "Ruff format"
			self._summary = "Improper formatting"
		elif result_rule_id:
			check_group = ""
			check_name = ""
			self._summary = f"{result_rule_id}: {self._summary}"
			check_group = "check"
			check_name = result_rule_id
			if rule_dict and result_rule_id in rule_dict:
				sarif_rule = rule_dict[result_rule_id]
				rule_props = sarif_rule.get("properties", {})
				rule_kind = rule_props.get("kind", None)
				rule_name = rule_props.get("name", None)
				if rule_kind:
					check_group = rule_kind
				if rule_name:
					check_name = rule_name
			if check_name or check_group:
				self._summary = f"{self._summary} [{check_group}:{check_name}]"
			self._title = f"Ruff {result_rule_id}"
		else:
			self._title = "Ruff check"

		self._suggestions = [RuffSuggestion(f) for f in sarif_result.get("fixes", [])]
		self._suggestions = [f for f in self._suggestions if f.changes]

	@property
	def level(self):
		"""Level of result for GHA workflow (error, warning, notice, debug)."""
		return self._level

	@property
	def summary(self):
		"""Main result message."""
		return self._summary

	@property
	def title(self):
		"""Result title for GHA workflow."""
		return self._title

	@property
	def locations(self):
		"""Result locations as mapping of pathlib Path objects to LineRange objects. None if result is unusable."""
		return self._locations

	@property
	def suggestions(self):
		"""List of suggested fixes as RuffSuggestion objects."""
		return self._suggestions

	def emit_gha_messages(self):
		"""
		Emit Github Action workflow messages and diffs of proposed fixes.

		Returns:
			True if no error or warning messages emitted, False otherwise.
		"""
		# Get message count
		msg_qty = 0
		for regions in self.locations.values():
			msg_qty += len(regions)

		print_msg_idx = msg_qty > 1
		if print_msg_idx:
			# Prep for message counter if more than one message in result
			msg_qty_str = f"{msg_qty}"
			msg_qty_str_len = len(msg_qty_str)
			msg_idx_fmt_str = "{0} {1:0" + msg_qty_str_len + "d}/" + msg_qty_str
		else:
			msg_qty_str = ""
			msg_qty_str_len = 0
			msg_idx_fmt_str = ""

		no_errors = True

		for file, regions in self.locations.items():
			for ridx, region in enumerate(regions):
				# Add message counter to title if more than one message in result
				title = msg_idx_fmt_str.format(self.title, ridx + 1) if print_msg_idx else self.title
				# create props dict
				message_props = OrderedDict([
					("title", title),
					("file", str(file)),
				])
				if region is not None:
					message_props["line"] = region.start
					if region.start_col:
						message_props["col"] = region.start_col
					if region.end:
						message_props["endLine"] = region.end
					if region.end_col:
						message_props["endColumn"] = region.end_col
				emit_gha_message(self.level, message_props, self.summary)
				if no_errors:
					no_errors = self.level not in {"error", "warning"}
		for suggestion in self.suggestions:
			print(suggestion.generate_diff())

		return no_errors


class RuffRun:
	"""Parsed sarif run from Ruff."""

	rules = {}

	def __init__(self, sarif_run, git_changes=None):
		"""
		Initialize RuffRun from sarif run.

		Args:
			sarif_run: Parsed json of a sarif run.
			git_changes: Mapping of pathlib Path objects to LineRange objects as returned by git_diff. Optional.
		"""
		# Get rules.
		RuffRun.rules |= {r["id"]: r for r in sarif_run.get("tool", {}).get("driver", {}).get("rules", [])}

		# Get results. Ignore results with kind of "notApplicable".
		self._results = [r for r in sarif_run.get("results", []) if r.get("kind", None) != "notApplicable"]
		self._results = [RuffResult(r, rule_dict=RuffRun.rules, all_git_changes=git_changes) for r in self._results]
		self._results = [r for r in self._results if r.locations] # Filter out unusable results.

	@property
	def results(self):
		"""List of results as RuffResult objects."""
		return self._results


def lint_check(args):
	"""
	Run ruff check, filter results, and emit messages.

	Args:
		args: Arguments passed to script (argparse).

	Returns:
		Intended return code for script. 0 if no warnings or errors were emitted, -1 otherwise.
	"""
	git_changes = git_diff(args.since_commit)
	check_files = get_check_files(git_changes)

	retcode = 0

	for check_file in check_files:
		checks_sarif = invoke_check(check_file)

		runs = [RuffRun(r, git_changes=git_changes) for r in checks_sarif.get("runs", [])]
		runs = [r for r in runs if r.results] # Filter out runs with no results.
		for run in runs:
			for result in run.results:
				res = result.emit_gha_messages()
				if not res:
					retcode = -1

	return retcode


def lint_format(args):
	"""
	Run ruff format, filter results, and emit messages.

	Args:
		args: Arguments passed to script (argparse).

	Returns:
		Intended return code for script. 0 if no warnings or errors were emitted, -1 otherwise.
	"""
	git_changes = git_diff(args.since_commit)
	check_files = get_check_files(git_changes)

	retcode = 0

	for check_file in check_files:
		for check_range in git_changes.get(check_file, []):
			format_sarif = invoke_format(check_file, check_range)

			runs = [RuffRun(r) for r in format_sarif.get("runs", [])]
			runs = [r for r in runs if r.results] # Filter out runs with no results.
			for run in runs:
				for result in run.results:
					res = result.emit_gha_messages()
					if not res:
						retcode = -1

	return retcode


def _add_common_arguments(argparser):
	argparser.add_argument(
		"--since-commit",
		help="Commit/ref to diff against, up to HEAD. Will run from HEAD to the working tree if not specified.",
		metavar="COMMIT",
		default=None,
	)


def _main():
	argparser = argparse.ArgumentParser()
	subparsers = argparser.add_subparsers(dest="cmd")

	check_argparser = subparsers.add_parser("check")
	_add_common_arguments(check_argparser)

	format_argparser = subparsers.add_parser("format")
	_add_common_arguments(format_argparser)

	args = argparser.parse_args()
	if args.cmd == "check":
		return lint_check(args)
	if args.cmd == "format":
		return lint_format(args)
	raise RuntimeError("Unrecognized command")

	return 0


if __name__ == "__main__":
	sys.exit(_main())
