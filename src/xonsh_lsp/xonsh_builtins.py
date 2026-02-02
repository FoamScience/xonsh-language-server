"""
Xonsh builtins and syntax documentation.

This module contains documentation and metadata for xonsh-specific
constructs, operators, builtins, and magic variables.
"""

# Xonsh subprocess operators
XONSH_OPERATORS = {
    "$(": {
        "doc": "Captured subprocess - executes command and returns stdout as a string.",
        "syntax": "$(command arg1 arg2)",
        "examples": [
            'output = $(ls -la)',
            'files = $(find . -name "*.py")',
        ],
    },
    "!(": {
        "doc": "Captured subprocess object - executes command and returns a CommandPipeline object with stdout, stderr, return code, etc.",
        "syntax": "!(command arg1 arg2)",
        "examples": [
            'result = !(git status)',
            'if !(grep -q pattern file).returncode == 0:',
        ],
    },
    "$[": {
        "doc": "Uncaptured subprocess - executes command, streams output to stdout/stderr.",
        "syntax": "$[command arg1 arg2]",
        "examples": [
            '$[make build]',
            '$[python script.py]',
        ],
    },
    "![": {
        "doc": "Uncaptured subprocess object - executes command, streams output, returns CommandPipeline.",
        "syntax": "![command arg1 arg2]",
        "examples": [
            '![npm install]',
            'proc = ![long_running_command]',
        ],
    },
    "@(": {
        "doc": "Python evaluation - evaluates Python expression and inserts result into subprocess.",
        "syntax": "@(python_expression)",
        "examples": [
            '$(echo @(filename))',
            '$(mv @(old_name) @(new_name))',
        ],
    },
    "@$(": {
        "doc": "Tokenized substitution - executes command and splits result into tokens.",
        "syntax": "@$(command)",
        "examples": [
            'for f in @$(ls):',
            'args = @$(echo one two three)',
        ],
    },
    "|": {
        "doc": "Pipe operator - connects stdout of left command to stdin of right command.",
        "syntax": "command1 | command2",
        "examples": [
            '$(cat file | grep pattern)',
            '$(ps aux | grep python)',
        ],
    },
    "a|": {
        "doc": "Pipe stdout and stderr - connects both streams to next command. Alias: all|",
        "syntax": "command1 a| command2",
        "examples": [
            '$(make a| tee build.log)',
            '$(cmd all| grep error)',
        ],
    },
    "e|": {
        "doc": "Pipe stderr only - connects stderr to next command. Alias: err|",
        "syntax": "command1 e| command2",
        "examples": [
            '$(make e| grep error)',
            '$(cmd err| logger)',
        ],
    },
    ">": {
        "doc": "Redirect stdout to file (overwrite).",
        "syntax": "command > file",
        "examples": [
            '$(echo hello > output.txt)',
        ],
    },
    ">>": {
        "doc": "Redirect stdout to file (append).",
        "syntax": "command >> file",
        "examples": [
            '$(echo line >> log.txt)',
        ],
    },
    "2>": {
        "doc": "Redirect stderr to file.",
        "syntax": "command 2> file",
        "examples": [
            '$(make 2> errors.txt)',
        ],
    },
    "&>": {
        "doc": "Redirect both stdout and stderr to file.",
        "syntax": "command &> file",
        "examples": [
            '$(build.sh &> build.log)',
        ],
    },
    "<": {
        "doc": "Redirect stdin from file.",
        "syntax": "command < file",
        "examples": [
            '$(sort < unsorted.txt)',
        ],
    },
    "&&": {
        "doc": "Logical AND - run right command only if left succeeds.",
        "syntax": "command1 && command2",
        "examples": [
            '$[make && make install]',
        ],
    },
    "||": {
        "doc": "Logical OR - run right command only if left fails.",
        "syntax": "command1 || command2",
        "examples": [
            '$[test -f file || touch file]',
        ],
    },
    "&": {
        "doc": "Background execution - run command in background.",
        "syntax": "command &",
        "examples": [
            '$(long_running_command &)',
        ],
    },
}

# Xonsh builtin functions
XONSH_BUILTINS = {
    "aliases": {
        "signature": "aliases",
        "doc": "Dictionary of command aliases. Can be modified to add/remove aliases.",
        "snippet": "aliases",
    },
    "source": {
        "signature": "source(filename)",
        "doc": "Execute xonsh script in current context, similar to bash source/dot command.",
        "snippet": 'source("${1:filename}")',
    },
    "xontrib": {
        "signature": "xontrib load name [name ...]",
        "doc": "Load xonsh extensions (xontribs).",
        "snippet": 'xontrib load ${1:name}',
    },
    "xonfig": {
        "signature": "xonfig [subcommand]",
        "doc": "Xonsh configuration utility. Subcommands: info, wizard, styles, colors.",
        "snippet": "xonfig ${1|info,wizard,styles,colors|}",
    },
    "cd": {
        "signature": "cd [path]",
        "doc": "Change current directory. Supports ~, -, and OLDPWD.",
        "snippet": "cd ${1:path}",
    },
    "pushd": {
        "signature": "pushd [path]",
        "doc": "Push directory onto stack and change to it.",
        "snippet": "pushd ${1:path}",
    },
    "popd": {
        "signature": "popd",
        "doc": "Pop directory from stack and change to it.",
        "snippet": "popd",
    },
    "dirs": {
        "signature": "dirs [-c] [-p] [-v]",
        "doc": "Display directory stack.",
        "snippet": "dirs",
    },
    "jobs": {
        "signature": "jobs",
        "doc": "Display background jobs.",
        "snippet": "jobs",
    },
    "fg": {
        "signature": "fg [job_id]",
        "doc": "Bring background job to foreground.",
        "snippet": "fg ${1:job_id}",
    },
    "bg": {
        "signature": "bg [job_id]",
        "doc": "Continue job in background.",
        "snippet": "bg ${1:job_id}",
    },
    "disown": {
        "signature": "disown [job_id]",
        "doc": "Remove job from job table.",
        "snippet": "disown ${1:job_id}",
    },
    "history": {
        "signature": "history [subcommand]",
        "doc": "Xonsh history management. Subcommands: show, info, gc, diff.",
        "snippet": "history ${1|show,info,gc,diff|}",
    },
    "trace": {
        "signature": "trace on|off",
        "doc": "Enable/disable command tracing for debugging.",
        "snippet": "trace ${1|on,off|}",
    },
    "timeit": {
        "signature": "timeit [command]",
        "doc": "Time command execution.",
        "snippet": "timeit ${1:command}",
    },
    "scp": {
        "signature": "scp [options] source dest",
        "doc": "Secure copy (wrapper with xonsh path expansion).",
        "snippet": "scp ${1:source} ${2:dest}",
    },
    "ssh": {
        "signature": "ssh [options] host",
        "doc": "SSH client (wrapper with xonsh integration).",
        "snippet": "ssh ${1:host}",
    },
}

# Xonsh aliases (common default aliases)
XONSH_ALIASES = {
    "ll": {
        "description": "ls -la",
        "doc": "Long listing with hidden files.",
    },
    "la": {
        "description": "ls -a",
        "doc": "List all files including hidden.",
    },
    "grep": {
        "description": "grep --color=auto",
        "doc": "Grep with color highlighting.",
    },
    "ls": {
        "description": "ls --color=auto",
        "doc": "List with color highlighting.",
    },
}

# Xonsh magic/special environment variables
XONSH_MAGIC_VARS = {
    "__xonsh__": {
        "type": "XonshSession",
        "doc": "The current xonsh session object.",
    },
    "PROMPT": {
        "type": "str | callable",
        "doc": "The primary prompt string or callable.",
    },
    "RPROMPT": {
        "type": "str | callable",
        "doc": "Right-side prompt string or callable.",
    },
    "TITLE": {
        "type": "str | callable",
        "doc": "Terminal window title.",
    },
    "MULTILINE_PROMPT": {
        "type": "str",
        "doc": "Prompt for continuation lines.",
    },
    "XONSH_HISTORY_SIZE": {
        "type": "int | tuple",
        "doc": "Maximum history size. Can be (size, 'commands') or (size, 'bytes').",
    },
    "XONSH_HISTORY_FILE": {
        "type": "str",
        "doc": "Path to history file.",
    },
    "XONSH_HISTORY_BACKEND": {
        "type": "str",
        "doc": "History backend: 'json', 'sqlite', or custom.",
    },
    "XONSH_COLOR_STYLE": {
        "type": "str",
        "doc": "Color style for syntax highlighting.",
    },
    "XONSH_SHOW_TRACEBACK": {
        "type": "bool",
        "doc": "Show full tracebacks on errors.",
    },
    "XONSH_TRACEBACK_LOGFILE": {
        "type": "str",
        "doc": "File to log tracebacks.",
    },
    "AUTO_CD": {
        "type": "bool",
        "doc": "Change directory by typing path without cd.",
    },
    "AUTO_PUSHD": {
        "type": "bool",
        "doc": "Automatically push directories on cd.",
    },
    "AUTO_SUGGEST": {
        "type": "bool",
        "doc": "Enable fish-like auto-suggestions.",
    },
    "AUTO_SUGGEST_IN_COMPLETIONS": {
        "type": "bool",
        "doc": "Include auto-suggest in completions.",
    },
    "CASE_SENSITIVE_COMPLETIONS": {
        "type": "bool",
        "doc": "Case-sensitive tab completions.",
    },
    "COMPLETIONS_DISPLAY": {
        "type": "str",
        "doc": "How to display completions: 'single', 'multi', 'readline'.",
    },
    "COMPLETIONS_MENU_ROWS": {
        "type": "int",
        "doc": "Number of rows in completion menu.",
    },
    "COMPLETION_IN_THREAD": {
        "type": "bool",
        "doc": "Run completions in background thread.",
    },
    "DOTGLOB": {
        "type": "bool",
        "doc": "Include hidden files in glob patterns.",
    },
    "EXPAND_ENV_VARS": {
        "type": "bool",
        "doc": "Expand environment variables in strings.",
    },
    "FOREIGN_ALIASES_OVERRIDE": {
        "type": "bool",
        "doc": "Allow foreign shell aliases to override xonsh aliases.",
    },
    "GLOB_SORTED": {
        "type": "bool",
        "doc": "Sort glob results.",
    },
    "HISTCONTROL": {
        "type": "set",
        "doc": "History control options: 'ignoredups', 'ignoreerr', etc.",
    },
    "IGNOREEOF": {
        "type": "bool",
        "doc": "Ignore EOF (Ctrl-D) for shell exit.",
    },
    "INDENT": {
        "type": "str",
        "doc": "Indentation string for multiline input.",
    },
    "MOUSE_SUPPORT": {
        "type": "bool",
        "doc": "Enable mouse support in prompt_toolkit.",
    },
    "PATH": {
        "type": "EnvPath",
        "doc": "List of directories to search for commands.",
    },
    "PATHEXT": {
        "type": "EnvPath",
        "doc": "File extensions to try when searching for commands (Windows).",
    },
    "PRETTY_PRINT_RESULTS": {
        "type": "bool",
        "doc": "Pretty-print expression results.",
    },
    "PROMPT_TOOLKIT_COLOR_DEPTH": {
        "type": "str",
        "doc": "Color depth: 'DEPTH_1_BIT', 'DEPTH_4_BIT', 'DEPTH_8_BIT', 'DEPTH_24_BIT'.",
    },
    "PUSHD_MINUS": {
        "type": "bool",
        "doc": "Swap pushd +/- meanings.",
    },
    "PUSHD_SILENT": {
        "type": "bool",
        "doc": "Don't print directory stack after pushd/popd.",
    },
    "SHELL_TYPE": {
        "type": "str",
        "doc": "Shell type: 'prompt_toolkit', 'readline', 'random', 'best'.",
    },
    "SUBSEQUENCE_PATH_COMPLETION": {
        "type": "bool",
        "doc": "Enable subsequence matching in path completion.",
    },
    "SUGGEST_COMMANDS": {
        "type": "bool",
        "doc": "Suggest corrections for mistyped commands.",
    },
    "SUGGEST_MAX_NUM": {
        "type": "int",
        "doc": "Maximum number of suggestions.",
    },
    "SUGGEST_THRESHOLD": {
        "type": "int",
        "doc": "Threshold for command suggestions.",
    },
    "SUPPRESS_BRANCH_TIMEOUT_MESSAGE": {
        "type": "bool",
        "doc": "Suppress branch timeout messages in prompt.",
    },
    "UPDATE_COMPLETIONS_ON_KEYPRESS": {
        "type": "bool",
        "doc": "Update completions on every keypress.",
    },
    "UPDATE_OS_ENVIRON": {
        "type": "bool",
        "doc": "Update os.environ when xonsh env changes.",
    },
    "UPDATE_PROMPT_ON_KEYPRESS": {
        "type": "bool",
        "doc": "Update prompt on every keypress.",
    },
    "VC_BRANCH_TIMEOUT": {
        "type": "float",
        "doc": "Timeout for version control branch detection.",
    },
    "VC_GIT_INCLUDE_UNTRACKED": {
        "type": "bool",
        "doc": "Include untracked files in git status.",
    },
    "VI_MODE": {
        "type": "bool",
        "doc": "Enable vi editing mode.",
    },
    "VIRTUAL_ENV": {
        "type": "str",
        "doc": "Path to active Python virtual environment.",
    },
    "XDG_CONFIG_HOME": {
        "type": "str",
        "doc": "XDG config directory (default: ~/.config).",
    },
    "XDG_DATA_HOME": {
        "type": "str",
        "doc": "XDG data directory (default: ~/.local/share).",
    },
    "XONSHRC": {
        "type": "list",
        "doc": "List of xonsh RC files to load.",
    },
    "XONSH_APPEND_NEWLINE": {
        "type": "bool",
        "doc": "Append newline to command output.",
    },
    "XONSH_AUTOPAIR": {
        "type": "bool",
        "doc": "Auto-pair brackets and quotes.",
    },
    "XONSH_CACHE_EVERYTHING": {
        "type": "bool",
        "doc": "Cache all xonsh scripts.",
    },
    "XONSH_CACHE_SCRIPTS": {
        "type": "bool",
        "doc": "Cache compiled xonsh scripts.",
    },
    "XONSH_CAPTURE_ALWAYS": {
        "type": "bool",
        "doc": "Always capture subprocess output.",
    },
    "XONSH_CONFIG_DIR": {
        "type": "str",
        "doc": "Xonsh configuration directory.",
    },
    "XONSH_DATA_DIR": {
        "type": "str",
        "doc": "Xonsh data directory.",
    },
    "XONSH_DEBUG": {
        "type": "int",
        "doc": "Debug level (0-2).",
    },
    "XONSH_ENCODING": {
        "type": "str",
        "doc": "Default encoding for subprocess communication.",
    },
    "XONSH_ENCODING_ERRORS": {
        "type": "str",
        "doc": "Encoding error handling strategy.",
    },
    "XONSH_INTERACTIVE": {
        "type": "bool",
        "doc": "Whether xonsh is running interactively.",
    },
    "XONSH_LOGIN": {
        "type": "bool",
        "doc": "Whether xonsh is a login shell.",
    },
    "XONSH_PROC_FREQUENCY": {
        "type": "float",
        "doc": "Process polling frequency in seconds.",
    },
    "XONSH_STORE_STDOUT": {
        "type": "bool",
        "doc": "Store stdout in _ variable.",
    },
}

# Xonsh string prefixes
XONSH_STRING_PREFIXES = {
    "p": {
        "doc": "Path literal - creates a pathlib.Path object.",
        "example": 'p"/home/user/file.txt"',
    },
    "pf": {
        "doc": "Path literal with f-string formatting.",
        "example": 'pf"/home/{user}/file.txt"',
    },
    "pr": {
        "doc": "Raw path literal (no escape processing).",
        "example": 'pr"C:\\Users\\name"',
    },
    "pb": {
        "doc": "Bytes path literal.",
        "example": 'pb"/home/user"',
    },
}

# Xonsh glob patterns
XONSH_GLOB_PATTERNS = {
    "`pattern`": {
        "doc": "Regex glob - matches files using regex pattern.",
        "example": '`.*\\.py$`',
    },
    "g`pattern`": {
        "doc": "Standard glob pattern.",
        "example": 'g`*.txt`',
    },
}

# Xonsh @ object attributes
XONSH_AT_OBJECTS = {
    "env": {
        "doc": "Environment dictionary. Access environment variables as attributes.",
        "example": "@.env.HOME",
    },
    "imp": {
        "doc": "Import helper. Import modules as attributes without import statement.",
        "example": "@.imp.json.loads(data)",
    },
    "lastcmd": {
        "doc": "Last executed command result.",
        "example": "@.lastcmd.returncode",
    },
    "aliases": {
        "doc": "Aliases dictionary. Access command aliases.",
        "example": "@.aliases.ll",
    },
    "builtins": {
        "doc": "Xonsh builtins namespace.",
        "example": "@.builtins.source",
    },
}

# Common xontribs (extensions)
XONSH_XONTRIBS = {
    "abbrevs": {
        "doc": "Command abbreviations (like fish shell).",
    },
    "argcomplete": {
        "doc": "Tab completion using argcomplete library.",
    },
    "autojump": {
        "doc": "Integration with autojump directory navigator.",
    },
    "autovox": {
        "doc": "Automatic virtual environment activation.",
    },
    "bashisms": {
        "doc": "Bash-like syntax support.",
    },
    "chatgpt": {
        "doc": "ChatGPT integration.",
    },
    "coreutils": {
        "doc": "Python implementations of common Unix commands.",
    },
    "dalias": {
        "doc": "Dynamic aliases.",
    },
    "direnv": {
        "doc": "Integration with direnv.",
    },
    "django": {
        "doc": "Django development helpers.",
    },
    "docker_tabcomplete": {
        "doc": "Docker command completions.",
    },
    "dracula": {
        "doc": "Dracula color theme.",
    },
    "fzf-widgets": {
        "doc": "FZF fuzzy finder widgets.",
    },
    "gitinfo": {
        "doc": "Git repository information in prompt.",
    },
    "github_copilot": {
        "doc": "GitHub Copilot integration.",
    },
    "history_encrypt": {
        "doc": "Encrypted command history.",
    },
    "jedi": {
        "doc": "Jedi-based Python completions.",
    },
    "jupyter": {
        "doc": "Jupyter notebook integration.",
    },
    "mpl": {
        "doc": "Matplotlib integration.",
    },
    "onepath": {
        "doc": "Single path argument handling.",
    },
    "pdb": {
        "doc": "Python debugger integration.",
    },
    "powerline": {
        "doc": "Powerline prompt theme.",
    },
    "prompt_starship": {
        "doc": "Starship prompt integration.",
    },
    "pyenv": {
        "doc": "Pyenv integration.",
    },
    "readable-traceback": {
        "doc": "More readable Python tracebacks.",
    },
    "schedule": {
        "doc": "Task scheduling.",
    },
    "sh": {
        "doc": "POSIX shell compatibility.",
    },
    "term_integration": {
        "doc": "Terminal integration (title, notifications).",
    },
    "vox": {
        "doc": "Virtual environment management.",
    },
    "whole_word_jumping": {
        "doc": "Ctrl+arrows jump whole words.",
    },
    "z": {
        "doc": "Directory jumping (like z/zoxide).",
    },
    "zoxide": {
        "doc": "Zoxide directory jumper integration.",
    },
    "1password": {
        "doc": "1Password CLI integration.",
    },
}
