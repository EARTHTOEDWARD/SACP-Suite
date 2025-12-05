# Dark Theme Quick Reference

## For Developers

### Adding a New Page with Plotly Graphs

```python
# 1. Import the dark layout
from sacp_suite.ui.pages.common import PLOTLY_DARK_LAYOUT
import plotly.graph_objects as go

# 2. Create your figure
fig = go.Figure(data=your_traces)

# 3. Apply dark theme FIRST
fig.update_layout(**PLOTLY_DARK_LAYOUT)

# 4. Then add your custom layout
fig.update_layout(
    title="Your Title",
    height=500,
    xaxis_title="X",
    yaxis_title="Y"
)
```

### Adding a New DataTable

```python
# 1. Import the dark table style
from sacp_suite.ui.pages.common import dark_table_style
from dash import dash_table

# 2. Create your table with the style
dash_table.DataTable(
    id="my-table",
    data=your_data,
    columns=your_columns,
    **dark_table_style(),  # Apply dark theme
)
```

### Color Palette Reference

```css
/* Backgrounds */
--bg-primary: #000000      /* Pure black */
--bg-secondary: #0a0a0a    /* Subtle layer */
--bg-tertiary: #141414     /* Cards, sections */
--bg-hover: #1a1a1a        /* Hover states */

/* Accents */
--accent-primary: #00CED1   /* Cyan - main accent */
--accent-bright: #1DE9B6    /* Bright cyan */
--accent-dark: #008B8B      /* Dark cyan */

/* Text */
--text-primary: #F5F5F5     /* Main text (white) */
--text-secondary: #B0B0B0   /* Secondary text */
--text-muted: #808080       /* Muted text */

/* Borders */
--border-subtle: #222222
--border-medium: #333333
--border-strong: #444444

/* Semantic */
--success: #10B981          /* Green */
--warning: #F59E0B          /* Amber */
--error: #EF4444            /* Red */
```

### Common CSS Classes

```html
<!-- Cards -->
<div className="card">
  <h3>Card Title</h3>
  <p>Card content...</p>
</div>

<!-- Section Title -->
<h3 className="section-title">My Section</h3>

<!-- Primary Button -->
<button className="primary-btn">Click Me</button>

<!-- Grid Layout -->
<div className="section-grid">
  <div className="card">Item 1</div>
  <div className="card">Item 2</div>
</div>

<!-- Upload Box -->
<div className="upload-box">
  Drag and drop files here
</div>

<!-- Status Messages -->
<div className="status-text">Status: OK</div>
<div className="status-text error">Error message</div>

<!-- Validation States -->
<span className="check-pass">✓ Passed</span>
<span className="check-warn">⚠ Warning</span>
<span className="check-fail">✗ Failed</span>
```

### Spacing System (8px grid)

```css
--space-xs: 4px
--space-sm: 8px
--space-md: 16px   /* Most common */
--space-lg: 24px
--space-xl: 32px
--space-2xl: 48px
```

Usage:
```css
margin: var(--space-md);
padding: var(--space-lg) var(--space-xl);
gap: var(--space-sm);
```

### Border Radius

```css
--radius-sm: 4px
--radius-md: 8px    /* Standard buttons */
--radius-lg: 12px   /* Cards */
--radius-xl: 16px
--radius-full: 9999px  /* Pills/badges */
```

### Transitions

```css
--transition-fast: 150ms ease   /* Hover colors */
--transition-base: 200ms ease   /* Standard */
--transition-slow: 300ms ease   /* Complex animations */
```

## Plotly Graph Customization

### Trace Colors (Auto-applied)
1. White (#FFFFFF)
2. Cyan (#00CED1)
3. Bright Blue (#4A9EFF)
4. Bright Cyan/Green (#1DE9B6)
5. Medium Slate Blue (#7B68EE)
6. Deep Sky Blue (#00BFFF)
7. Medium Turquoise (#48D1CC)
8. Sky Blue (#87CEEB)

### Override Specific Elements

```python
# After applying PLOTLY_DARK_LAYOUT, you can override:
fig.update_layout(
    **PLOTLY_DARK_LAYOUT,
    title="My Custom Title",
    title_font_size=20,
    title_font_color="#00CED1",  # Cyan accent
    xaxis=dict(
        title="Custom X",
        gridcolor="#333333",  # Slightly brighter grid
    ),
    margin=dict(l=50, r=50, t=80, b=50),
)

# For 3D plots, the 'scene' is already configured
# But you can add more:
fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
)
```

## Common Patterns

### Loading States

```python
dcc.Loading(
    dcc.Graph(id="my-graph"),
    type="dot"  # Will use cyan spinner
)
```

### Inline Forms

```html
<div className="inline-form">
  <label>Parameter:</label>
  <input type="number" value="1.0" />
  <button className="primary-btn">Run</button>
</div>
```

### Status Display

```python
html.Div(
    id="status-output",
    className="status-text"
)

# In callback:
return "Operation completed successfully"  # Normal
return html.Span("Error occurred", className="error")  # Error
```

### Explainer Sections

```python
html.Details(
    [
        html.Summary("ℹ About this page"),
        html.Div(
            [
                html.P("Explanation text..."),
                html.Ul([
                    html.Li("Feature 1"),
                    html.Li("Feature 2"),
                ])
            ],
            className="explainer-lede"
        )
    ],
    className="explainer"
)
```

## Testing Your Changes

```bash
# 1. Start the server
cd "/Users/edward/SACP SUITE"
python -m sacp_suite.ui.app

# 2. Open browser to http://127.0.0.1:8050

# 3. Check your page:
#    - Dark backgrounds everywhere
#    - Cyan accents on interactive elements
#    - White/blue graph traces
#    - Smooth hover effects
#    - Good contrast on all text
```

## Troubleshooting

### Graph not dark?
```python
# Make sure you imported and applied PLOTLY_DARK_LAYOUT
from sacp_suite.ui.pages.common import PLOTLY_DARK_LAYOUT
fig.update_layout(**PLOTLY_DARK_LAYOUT)  # BEFORE other updates
```

### Table not dark?
```python
# Make sure you're using the helper function
from sacp_suite.ui.pages.common import dark_table_style
dash_table.DataTable(**dark_table_style(), ...)
```

### Element has wrong color?
```css
/* Check if you're overriding CSS variables */
/* Use browser dev tools to inspect computed styles */
/* Reference style.css for the correct variable */
```

### Hover effect not working?
```css
/* Make sure your element has the transition */
transition: all var(--transition-base);

/* And the hover state defined */
.my-element:hover {
  background: var(--bg-hover);
  color: var(--accent-primary);
}
```

## File Locations

- **Main stylesheet**: `src/sacp_suite/ui/assets/style.css`
- **Theme helpers**: `src/sacp_suite/ui/pages/common.py`
- **App shell**: `src/sacp_suite/ui/app.py`
- **Example page**: `src/sacp_suite/ui/pages/simulator.py`

## Need Help?

1. Read [DARK_THEME_SUMMARY.md](DARK_THEME_SUMMARY.md) for detailed explanation
2. Check [style.css](src/sacp_suite/ui/assets/style.css) for component examples
3. Look at [simulator.py](src/sacp_suite/ui/pages/simulator.py) for full implementation
4. Use browser DevTools to inspect styles

---

**Remember**: Always apply `PLOTLY_DARK_LAYOUT` BEFORE custom layout updates!
