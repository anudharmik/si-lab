"""
Comprehensive Streamlit Headless Test Runner
Validates that all 9 pages of app.py render without errors across various selections.
"""

import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_all_pages():
    print("Testing Streamlit App across all 9 pages...")
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()
    assert not at.exception, f"Overview page encountered exception: {at.exception}"
    print("  [OK] Overview page loaded cleanly.")
    
    pages = [
        "Historical Trends",
        "Statistics",
        "Return Distribution",
        "Sampling & CLT",
        "Hypothesis Testing",
        "Correlation Analysis",
        "Model Performance",
        "Predictions"
    ]
    
    for page in pages:
        # Find the navigation radio
        nav_radio = None
        for radio in at.sidebar.radio:
            if "Overview" in radio.options:
                nav_radio = radio
                break
        assert nav_radio is not None, "Navigation radio not found"
        
        # Select page and rerun
        nav_radio.set_value(page)
        at.run()
        assert not at.exception, f"Page '{page}' encountered exception: {at.exception}"
        print(f"  [OK] Page '{page}' rendered cleanly.")
        
    # Test changing stock selection to MSFT
    stock_select = at.sidebar.selectbox[0]
    stock_select.set_value("MSFT")
    at.run()
    assert not at.exception, f"Stock selection change encountered exception: {at.exception}"
    print("  [OK] Target Stock change (MSFT) succeeded.")
    
    # Test changing Date Window to 'Past 1 Year'
    for r in at.sidebar.radio:
        if "Past 1 Year" in r.options:
            r.set_value("Past 1 Year")
            break
    at.run()
    assert not at.exception, f"Date Window change encountered exception: {at.exception}"
    print("  [OK] Date window change ('Past 1 Year') succeeded.")
    
    # Test switching back to Overview with modified stock and date
    nav_radio.set_value("Overview")
    at.run()
    assert not at.exception, f"Overview with custom stock/date encountered exception: {at.exception}"
    print("  [OK] Overview with custom stock & date rendered cleanly.")

    print("\nALL PAGES, SELECTORS, AND INTERACTIVE CONTROLS PASSED HEADLESS VALIDATION!")

if __name__ == "__main__":
    test_all_pages()
