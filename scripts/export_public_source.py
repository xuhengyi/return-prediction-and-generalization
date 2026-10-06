"""Export an explicit set of analytical source files for public sharing."""
import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    '.gitignore', 'LICENSE', 'README.md', 'config.example.json',
    'requirements.txt', 'requirements-lock.txt', 'run.py',
    'references/kmz.bib', 'references/montanari_urbani.bib',
    'scripts/export_public_source.py', 'scripts/render_report.py',
    'scripts/reproduce.py', 'scripts/review_pages.py', 'scripts/validate.py',
    'src/__init__.py', 'src/core.py', 'src/dynamics.py', 'src/figures.py',
    'src/prediction.py', 'src/report.py', 'src/simulation.py',
    'tests/test_core.py', 'tests/test_schema.py',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List the exact public files without writing an archive')
    parser.add_argument('--output', type=Path, default=ROOT/'dist/project1-public-source.zip')
    args = parser.parse_args()
    for name in PUBLIC_FILES:
        path = ROOT/name
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            parser.error(f'Missing or symlinked public input: {name}')
    if args.list:
        print('\n'.join(PUBLIC_FILES))
        return
    target = args.output.resolve()
    if target.suffix.lower() != '.zip':
        parser.error('The output must be a .zip file')
    if target.exists():
        parser.error(f'Refusing to overwrite {target}; choose a new --output')
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in PUBLIC_FILES:
            archive.write(ROOT/name, arcname=f'project1/{name}')
    with zipfile.ZipFile(target) as archive:
        damaged = archive.testzip()
        if damaged:
            raise ValueError(f'Archive integrity check failed: {damaged}')
    print(f'Public source archive: {target} ({len(PUBLIC_FILES)} files, {target.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
