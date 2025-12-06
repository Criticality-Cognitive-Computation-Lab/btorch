import inspect
from pathlib import Path


TEST_PATH = Path(__file__).parent.parent.resolve()
FIG_PATH = TEST_PATH.parent.resolve() / "fig/tests"


def caller_file():
    # 0 == this frame, 1 == caller of this function
    frame = inspect.stack()[2]
    return frame.filename

def fig_path(file=None) -> Path:
    file = file or caller_file()
    path = FIG_PATH / Path(file).relative_to(TEST_PATH).with_suffix("")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig, name=None, path=None):
    if path is None:
        path = Path(fig_path(caller_file()))
    if name is None:
        name = Path(__file__).stem
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}/{name}.pdf", transparent=True)
