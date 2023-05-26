import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def downloader(url: str, path: str, file_size: int, resume_byte_pos: int = None):
    with Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as pbar:
        # Append information to resume download
        resume_header = (
            {"Range": f"bytes={resume_byte_pos}-"} if resume_byte_pos else None
        )

        # Establish connection
        r = requests.get(url, stream=True, headers=resume_header)

        # Set configuration
        block_size = 1024
        initial_pos = resume_byte_pos or 0
        mode = "ab" if resume_byte_pos else "wb"

        task_id = pbar.add_task(
            description=f"Downloading {path.name}",
            total=file_size,
            completed=initial_pos,
        )

        with open(path, mode) as f:
            for chunk in r.iter_content(32 * block_size):
                f.write(chunk)
                pbar.update(task_id, advance=len(chunk))


def download_file(url: str, path: str) -> None:
    """Execute the correct download operation.
    Depending on the size of the file online and offline, resume the
    download if the file offline is smaller than online.
    """
    # Establish connection to header of file
    r = requests.head(url)

    # Get filesize of online and offline file
    file_size_online = int(r.headers.get("content-length", 0))

    if path.exists():
        file_size_offline = path.stat().st_size

        if file_size_online != file_size_offline:
            # Resume download
            downloader(url, path, file_size_online, file_size_offline)
    else:
        # Download
        downloader(url, path, file_size_online)
