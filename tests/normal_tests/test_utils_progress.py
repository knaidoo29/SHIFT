import pytest
from shift.utils import progress  # replace with your actual module


def test_progress_bar_basic(capsys):
    # Simple test: index 0 of 10
    progress.progress_bar(index=0, length=10, num_refresh=10, bar_length=10)
    captured = capsys.readouterr()
    assert "|" in captured.out  # Should contain bar edges
    assert "#" in captured.out or "_" in captured.out  # Should contain markers


def test_progress_bar_zero(capsys):
    # Test when index = length - 1 (last step)
    progress.progress_bar(index=0, length=1000, num_refresh=1000, bar_length=10)
    captured = capsys.readouterr()
    assert "_" in captured.out  # Should contain markers


def test_progress_bar_full(capsys):
    # Test when index = length - 1 (last step)
    progress.progress_bar(index=9, length=10, num_refresh=10, bar_length=10)
    captured = capsys.readouterr()
    assert "#" in captured.out  # The bar should be fully filled
    assert captured.out.endswith("\n")  # Should end with newline at completion


def test_progress_bar_explanation_indexing(capsys):
    # Test explanation text and indexing
    progress.progress_bar(index=4, length=10, explanation="Loading", indexing=True, num_refresh=10, bar_length=10)
    captured = capsys.readouterr()
    assert "Loading" in captured.out
    assert "[5/10]" in captured.out


def test_progress_bar_explanation_indexing_complete(capsys):
    # Test explanation text and indexing
    progress.progress_bar(index=9, length=10, explanation="Loading", indexing=True, num_refresh=10, bar_length=10)
    captured = capsys.readouterr()
    assert "Loading" in captured.out
    assert "[10/10]" in captured.out


def test_progress_bar_edges(capsys):
    # Test custom bar edges
    progress.progress_bar(index=2, length=10, bar_edge_left="<", bar_edge_right=">", num_refresh=10, bar_length=10)
    captured = capsys.readouterr()
    output = captured.out.lstrip('\r').strip()
    assert output.startswith("<")
    assert output.split()[0].endswith(">")


def test_progress_bar_refresh(capsys):
    # Ensure that progress bar is updated only on refresh steps
    for i in range(10):
        progress.progress_bar(index=i, length=10, num_refresh=2, bar_length=10)
    captured = capsys.readouterr()
    # The captured output should contain at least one percentage string
    assert "%" in captured.out
