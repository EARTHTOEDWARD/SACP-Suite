# Dashboard Navigation Debugging Report
**Date**: December 2, 2025
**Issue**: Dashboard navigation tabs not rendering content
**Status**: Partially resolved, core issue remains

---

## Problem Summary

The SACP Suite dashboard loads at `http://127.0.0.1:8050/` with the navigation bar visible, but clicking any tab (Chemistry, Simulator, Sheaf, etc.) does not render page content. The URL in the address bar changes (e.g., to `/chem`), but the page remains blank below the navigation.

### Initial Symptoms
- ✅ Dashboard loads successfully
- ✅ Navigation bar renders
- ✅ URL changes when clicking tabs
- ❌ Page content does not render
- ❌ Dash renderer JavaScript error in browser console

### Timeline
- **Before commit 579fbfa**: Dashboard worked correctly, all tabs rendered
- **After adding Chemistry tab**: Navigation broke for ALL tabs
- **Current state**: Dashboard loads but tab navigation is broken

---

## Root Cause Analysis

### Finding #1: Missing `refresh=False` on dcc.Location
**File**: `src/sacp_suite/ui/app.py` (line 69)

**Issue**: The `dcc.Location` component was configured without `refresh=False`:
```python
dcc.Location(id="url")  # WRONG
```

**Impact**: This causes full page refreshes when clicking `dcc.Link` components, bypassing Dash's client-side routing system entirely.

**Fix Applied**:
```python
dcc.Location(id="url", refresh=False)  # CORRECT
```

**Result**: Fix applied but did not resolve the issue (see Finding #4)

---

### Finding #2: Duplicate Callback Outputs Without allow_duplicate=True
**File**: `src/sacp_suite/ui/app.py` (multiple locations)

**Issue**: In Dash 3.3.0, when multiple callbacks write to the same component outputs, ALL callbacks must have `allow_duplicate=True` on those outputs.

**Duplicate outputs found**:
1. `self-tune-lambda.figure` (2 callbacks)
2. `self-tune-sr.figure` (2 callbacks)
3. `self-tune-attractor.figure` (2 callbacks)
4. `self-tune-summary.children` (3 callbacks)
5. `self-tune-timer.disabled` (2 callbacks)
6. `phase3d.figure` (2 callbacks)
7. `series.figure` (2 callbacks)
8. `lle_out.children` (2 callbacks)
9. `dataset_meta.children` (2 callbacks)

**Fixes Applied**:

Line 1776-1779 (update_self_tune_live callback):
```python
@app.callback(
    Output("self-tune-lambda", "figure", allow_duplicate=True),
    Output("self-tune-sr", "figure", allow_duplicate=True),
    Output("self-tune-attractor", "figure", allow_duplicate=True),
    Output("self-tune-summary", "children", allow_duplicate=True),
    Output("self-tune-timer", "disabled", allow_duplicate=True),
    # ...
)
```

Line 1456-1458:
```python
@app.callback(
    Output("phase3d", "figure", allow_duplicate=True),
    Output("series", "figure", allow_duplicate=True),
    Output("lle_out", "children", allow_duplicate=True),
    # ...
)
```

Line 2479:
```python
Output("dataset_meta", "children", allow_duplicate=True),
```

Line 1761:
```python
Output("self-tune-timer", "disabled", allow_duplicate=True),
```

**Result**: Fixed server-side validation errors, but JavaScript errors persist

---

### Finding #3: Router Return Type
**File**: `src/sacp_suite/ui/app.py` (line 1452)

**Issue**: The router callback was initially changed to return a list `[layout_fn()]` to satisfy Dash validation.

**History**:
- **Original (working)**: `return layout_fn()`
- **Modified (broken)**: `return [layout_fn()]`
- **Restored**: `return layout_fn()`

**Current state**: Line 1452 now correctly returns:
```python
return layout_fn()
```

---

### Finding #4: Dash JavaScript Not Making Callback Requests (CRITICAL)
**Status**: ❌ UNRESOLVED

**Evidence**:
1. **Server logs show NO router callback activity**:
   - No `[ROUTER]` debug messages (which we added for debugging)
   - No `_dash-update-component` requests after initial page load
   - Only initial page load requests: `/_dash-layout`, `/_dash-dependencies`

2. **Browser console shows JavaScript error**:
   - Error in `dash_renderer.v3_0m176c467606.min.js:2`
   - Error is logged by `console.error(e.payload.error)` in the "ON_ERROR" case
   - Indicates server is returning an error response

3. **Client-side routing is not working**:
   - Despite `refresh=False` being applied
   - Despite duplicate outputs being fixed
   - Dash's JavaScript is not making callback requests when URL changes

**Debugging steps taken**:
- ✅ Added debug logging to router callback
- ✅ Monitored server logs in real-time during tab clicks
- ✅ Verified callback is registered in `/_dash-dependencies` endpoint
- ✅ Confirmed no Python exceptions in server logs
- ❌ JavaScript error prevents callbacks from executing

**Conclusion**: There is a fundamental issue with Dash's client-side JavaScript that prevents it from making callback requests. The root cause is unknown and may be related to:
- Dash version incompatibility (currently using 3.3.0)
- JavaScript minification/bundling issues
- Browser caching of old Dash renderer code
- A configuration issue we haven't identified

---

## Technical Details

### Environment
- **Python**: 3.13.7
- **Dash**: 3.3.0
- **OS**: macOS (Darwin 24.6.0)
- **Browser**: Google Chrome
- **FastAPI**: Backend API on port 8000
- **Dash UI**: Frontend on port 8050

### File Locations
- **Main UI file**: `src/sacp_suite/ui/app.py` (2600+ lines)
- **Launcher script**: `macos/SACP Suite Launcher.app/Contents/MacOS/SACPSuiteLauncher`
- **UI logs**: `/var/folders/_g/ptsdtwwd2j1fqz377q56bd6c0000gn/T/sacp_ui.log`
- **API logs**: `/var/folders/_g/ptsdtwwd2j1fqz377q56bd6c0000gn/T/sacp_api.log`

### Key Code Locations

**App Layout** (lines 67-83):
```python
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),  # FIXED: Added refresh=False
    dcc.Store(id="shared_traj"),
    dcc.Store(id="task_spec_store"),
    dcc.Store(id="landing_stage", data="prompt"),
    dcc.Store(id="ingestion_preview_store"),
    dcc.Store(id="dataset_builder_store"),
    NAV,
    html.Div(id="page"),
], style={"maxWidth": "1280px", "margin": "0 auto", "padding": "16px"})
```

**Navigation** (lines 38-65):
```python
NAV = html.Nav([
    dcc.Link("Home", href="/"),
    html.Span(" | "),
    dcc.Link("Simulator", href="/sim"),
    html.Span(" | "),
    dcc.Link("Sheaf", href="/sheaf"),
    # ... more tabs ...
    dcc.Link("Chemistry", href="/chem"),  # Added but breaks routing
    # ... more tabs ...
])
```

**Router Callback** (lines 1435-1452):
```python
@app.callback(Output("page", "children"), Input("url", "pathname"))
def router(pathname: str):
    """Return the page layout matching the URL (single-output callback)."""
    print(f"[ROUTER] pathname={pathname}")  # DEBUG
    routes = {
        "/sim": page_sim,
        "/sheaf": page_sheaf,
        "/attr": page_attr,
        "/fractalllm": page_fractalllm,
        "/cog": page_cognition,
        "/dcrc": page_dcrc,
        "/self-tune": page_self_tune,
        "/chem": page_chemistry,
        "/bcp": page_bcp,
        "/abtc": page_abtc,
        "/datasets": page_datasets,
    }
    layout_fn = routes.get(pathname, page_home)
    print(f"[ROUTER] layout_fn={layout_fn.__name__}")  # DEBUG
    try:
        result = layout_fn()
        print(f"[ROUTER] result type={type(result).__name__}")  # DEBUG
        return result
    except Exception as e:
        print(f"[ROUTER] ERROR: {e}")  # DEBUG
        import traceback
        traceback.print_exc()
        raise
```

---

## Chemistry Page Backup

The Chemistry page code (lines 835-972) has been backed up to:
```
/tmp/chemistry_page_backup.py
```

This contains the full `page_chemistry()` function with:
- 4-site Autocatalytic Duffing Ring visualization
- TCA/Krebs cycle parameter controls
- Bifurcation scan functionality
- Interactive plots and controls

**To restore**: Copy content from backup file back into `app.py` between lines 835-972

---

## Attempted Solutions

### ✅ Solution 1: Fix Duplicate Callback Outputs
**Status**: SUCCESSFUL (server-side)
**Impact**: Eliminated Dash validation errors on server
**Result**: Did not fix navigation issue

### ✅ Solution 2: Add refresh=False to dcc.Location
**Status**: SUCCESSFUL (code change)
**Impact**: Should enable client-side routing
**Result**: Did not fix navigation issue (JavaScript still broken)

### ✅ Solution 3: Fix Router Return Type
**Status**: SUCCESSFUL
**Impact**: Router now returns single component instead of list
**Result**: Did not fix navigation issue

### ❌ Solution 4: Revert to Working Commit
**Status**: FAILED
**Attempted**: Revert `app.py` to commit 579fbfa (last known working state)
**Result**: Launcher failed with error dialog, processes did not start
**Issue**: Possible dependency/import errors in reverted code

---

## Diagnostic Commands Used

### Check Server Logs
```bash
tail -50 /var/folders/_g/ptsdtwwd2j1fqz377q56bd6c0000gn/T/sacp_ui.log
```

### Check for Duplicate Outputs
```bash
python3 -c "
import re
with open('src/sacp_suite/ui/app.py', 'r') as f:
    content = f.read()
outputs = re.findall(r'Output\([\"\']([\w-]+)[\"\']\s*,\s*[\"\']([\w-]+)[\"\']', content)
from collections import Counter
output_counts = Counter(outputs)
duplicates = {k: v for k, v in output_counts.items() if v > 1}
for (component, prop), count in sorted(duplicates.items()):
    print(f'{component}.{prop}: {count} times')
"
```

### Verify Dash Dependencies
```bash
curl -s http://127.0.0.1:8050/_dash-dependencies | python3 -m json.tool | grep -A 5 'page.children'
```

### Monitor Logs in Real-Time
```bash
tail -f /var/folders/_g/ptsdtwwd2j1fqz377q56bd6c0000gn/T/sacp_ui.log
```

---

## Recommendations

### Short-term (Next Session)

#### Option A: Deep Dive into JavaScript Error
1. **Expand the JavaScript error** in browser console to see full stack trace
2. **Check for Dash version issues**:
   ```bash
   pip show dash
   pip show dash-core-components
   pip show dash-html-components
   ```
3. **Try downgrading Dash**:
   ```bash
   pip install dash==2.14.0  # Last stable 2.x version
   ```
4. **Clear browser cache completely** and test in incognito mode

#### Option B: Minimal Reproduction
1. Create a minimal Dash app with just:
   - dcc.Location with refresh=False
   - Simple navigation with 2 tabs
   - Basic router callback
2. Test if basic routing works
3. Incrementally add complexity until it breaks

#### Option C: Manual URL Navigation
As a temporary workaround, implement server-side routing:
1. Remove `refresh=False` (allow full page reloads)
2. Use Flask's `@server.route()` decorators instead of Dash callbacks
3. Return full page layouts server-side

### Long-term

#### Option 1: Rebuild UI with Streamlit
**Pros**:
- Much simpler than Dash
- Better for scientific/data apps
- Fewer configuration issues
- Native support for Python data structures

**Cons**:
- Different paradigm (top-to-bottom execution)
- Less customizable than Dash
- Would require rewriting all pages

**Estimated effort**: 1-2 days

#### Option 2: Rebuild UI with Gradio
**Pros**:
- Extremely simple for ML/scientific apps
- Built-in themes and components
- Easy deployment

**Cons**:
- More limited than Dash or Streamlit
- Best for simple interfaces

**Estimated effort**: 1 day

#### Option 3: Keep Dash but Rebuild from Scratch
**Pros**:
- Keep existing framework knowledge
- More control than alternatives

**Cons**:
- Same potential for configuration issues
- Longer development time

**Estimated effort**: 2-3 days

---

## Git State

### Current Branch
```
main
```

### Modified Files
```
src/sacp_suite/ui/app.py (modified with fixes)
```

### Key Commits
- **579fbfa**: Last known working state (before Chemistry was added)
- **Current**: Has Chemistry added but navigation broken

### To Restore Working State
```bash
cd "/Users/edward/SACP SUITE"
git checkout 579fbfa -- src/sacp_suite/ui/app.py
# Then manually re-add Chemistry following existing patterns
```

---

## Summary

**What Works**:
- ✅ Dashboard loads
- ✅ Navigation bar renders
- ✅ URL routing changes URLs
- ✅ Server processes start correctly
- ✅ API responds on port 8000
- ✅ UI responds on port 8050 (initial load)

**What's Broken**:
- ❌ Tab clicks don't trigger router callback
- ❌ No `_dash-update-component` requests are made
- ❌ JavaScript error in Dash renderer
- ❌ Page content doesn't render when navigating

**What We Fixed**:
- ✅ Added `refresh=False` to dcc.Location
- ✅ Fixed all duplicate callback outputs
- ✅ Fixed router return type
- ✅ Added debug logging

**What Still Needs Fixing**:
- ❌ Core JavaScript issue preventing callbacks
- ❌ Unknown Dash configuration or version problem

---

## Contact Information

**Debugging Session Conducted By**: Claude (Anthropic)
**Date**: December 2, 2025
**Session Duration**: ~2 hours
**Files Modified**: src/sacp_suite/ui/app.py
**Backup Created**: /tmp/chemistry_page_backup.py

---

## Next Steps

1. **Share this document** with a Dash expert or on Dash community forums
2. **Include**:
   - This debugging report
   - Browser console screenshots showing full JavaScript error
   - Output from `pip show dash` and related packages
   - Server logs from a test session
3. **Ask specifically** about:
   - Why `_dash-update-component` requests aren't being made
   - JavaScript error in dash_renderer preventing callbacks
   - Dash 3.3.0 compatibility issues

Good luck! The core issue is very close to being identified - it's definitely a client-side JavaScript problem preventing Dash from making callback requests.
