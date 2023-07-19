def setup_print_hooks():
    import os
    if not os.environ.get("IFIELD_PRETTY_TRACEBACK", None):
        return

    from rich.traceback import install
    from rich.console import Console
    import warnings, sys

    if not os.isatty(2):
        # https://github.com/Textualize/rich/issues/1809
        os.environ.setdefault("COLUMNS", "120")

    install(
        show_locals = bool(os.environ.get("SHOW_LOCALS", "")),
        width       = None,
    )

    # custom warnings
    # https://github.com/Textualize/rich/issues/433

    from rich.traceback import install
    from rich.console import Console
    import warnings, sys


    def showwarning(message, category, filename, lineno, file=None, line=None):
        msg = warnings.WarningMessage(message, category, filename, lineno, file, line)

        if file is None:
            file = sys.stderr
            if file is None:
                # sys.stderr is None when run with pythonw.exe:
                # warnings get lost
                return
        text = warnings._formatwarnmsg(msg)
        if file.isatty():
            Console(file=file, stderr=True).print(text)
        else:
            try:
                file.write(text)
            except OSError:
                # the file (probably stderr) is invalid - this warning gets lost.
                pass
    warnings.showwarning = showwarning

    def warning_no_src_line(message, category, filename, lineno, file=None, line=None):
        if (file or sys.stderr) is not None:
            if (file or sys.stderr).isatty():
                if file is None or file is sys.stderr:
                    return f"[yellow]{category.__name__}[/yellow]: {message}\n    ({filename}:{lineno})"
        return f"{category.__name__}: {message} ({filename}:{lineno})\n"
    warnings.formatwarning = warning_no_src_line


setup_print_hooks()
del setup_print_hooks
