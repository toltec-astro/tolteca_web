var dagcomponentfuncs = (window.dashAgGridComponentFunctions =
    window.dashAgGridComponentFunctions || {});

/**
 * RoachRenderer — colored squares showing per-roach presence (Raw Obs tab).
 * Cell value: JSON string {"present": [...], "zarr": [...]}
 */
dagcomponentfuncs.RoachRenderer = function (props) {
    var val = props.value;
    if (!val) return null;
    var d;
    try { d = JSON.parse(val); } catch (e) { return React.createElement('span', null, val); }

    var present = d.present || [];
    var zarr    = d.zarr    || [];
    var groups  = [[0,1,2,3,4,5,6], [7,8,9,10], [11,12]];
    var colors  = ['#228be6', '#12b886', '#7950f2'];

    var children = [];
    groups.forEach(function (g, gi) {
        if (gi > 0) {
            children.push(React.createElement('span', {
                key: 'dot' + gi,
                style: { color: '#aaa', margin: '0 3px', fontSize: '9px' }
            }, '\u00b7'));
        }
        g.forEach(function (nw) {
            var isP = present.indexOf(nw) >= 0;
            var isZ = zarr.indexOf(nw) >= 0;
            children.push(React.createElement('span', {
                key: nw,
                title: 'toltec' + nw + (isZ ? ' \u2713' : ''),
                style: {
                    display: 'inline-block',
                    width: '11px',
                    height: '11px',
                    borderRadius: '2px',
                    background: isP ? colors[gi] : '#dee2e6',
                    opacity: (isP && !isZ) ? '0.45' : '1',
                }
            }));
        });
    });

    return React.createElement('div', {
        style: { display: 'flex', gap: '1px', alignItems: 'center', height: '100%' }
    }, children);
};

/**
 * TypeChip — colored badge for dp_type (All DP tab).
 * props.value: dp_type string e.g. "raw_obs", "cal_groups"
 */
dagcomponentfuncs.TypeChip = function (props) {
    var colorMap = {
        raw_obs:    { bg: 'rgba(34,139,230,0.15)',  text: '#1971c2' },
        cal_groups: { bg: 'rgba(64,192,87,0.15)',   text: '#2f9e44' },
        drivefit:   { bg: 'rgba(253,126,20,0.15)',  text: '#e8590c' },
        focus:      { bg: 'rgba(240,101,149,0.15)', text: '#c2255c' },
        astig:      { bg: 'rgba(190,75,219,0.15)',  text: '#9c36b5' },
        oof:        { bg: 'rgba(92,124,250,0.15)',  text: '#3b5bdb' },
    };
    var labelMap = {
        raw_obs: 'RAW OBS', cal_groups: 'CAL GRP',
        drivefit: 'DRIVEFIT', focus: 'FOCUS',
        astig: 'ASTIG', oof: 'OOF',
    };
    var c = colorMap[props.value] || { bg: '#f1f3f5', text: '#495057' };
    var label = labelMap[props.value] || String(props.value || '');
    return React.createElement('span', {
        style: {
            background: c.bg, color: c.text,
            borderRadius: '4px', padding: '2px 7px',
            fontSize: '11px', fontWeight: '700', letterSpacing: '0.3px',
            whiteSpace: 'nowrap',
        }
    }, label);
};

/**
 * KindChip — colored chip for data kind (All DP tab).
 * Uses props.data.kind_label, kind_color_hex, kind_tooltip.
 */
dagcomponentfuncs.KindChip = function (props) {
    var d = props.data || {};
    var label = d.kind_label || '—';
    var color = d.kind_color_hex || '#868e96';
    var tooltip = d.kind_tooltip || null;
    if (!label || label === '—') {
        return React.createElement('span', { style: { color: '#aaa' } }, '—');
    }
    // Parse hex to add ~15% opacity background
    return React.createElement('span', {
        title: tooltip || undefined,
        style: {
            background: color + '26',
            color: color,
            borderRadius: '4px', padding: '2px 7px',
            fontSize: '11px', fontWeight: '700',
            whiteSpace: 'nowrap',
            cursor: tooltip ? 'help' : 'default',
        }
    }, label);
};

/**
 * ObsnumsCell — first obsnum + count badge for groups (All DP tab).
 * Uses props.data.obsnums_first, obsnums_count, obsnums_tooltip.
 */
dagcomponentfuncs.ObsnumsCell = function (props) {
    var d = props.data || {};
    var first = d.obsnums_first;
    var count = d.obsnums_count || 1;
    var tooltip = d.obsnums_tooltip || null;
    if (first === null || first === undefined) {
        return React.createElement('span', { style: { color: '#aaa' } }, '—');
    }
    if (count <= 1) {
        return React.createElement('span', {
            style: { fontFamily: 'monospace', fontSize: '12px' }
        }, String(first));
    }
    return React.createElement('span', {
        title: tooltip || undefined,
        style: { display: 'flex', gap: '5px', alignItems: 'center', cursor: 'help' }
    }, [
        React.createElement('span', {
            key: 'f',
            style: { fontFamily: 'monospace', fontSize: '12px' }
        }, String(first)),
        React.createElement('span', {
            key: 'b',
            style: {
                background: '#f1f3f5', color: '#868e96',
                borderRadius: '4px', padding: '1px 5px',
                fontSize: '11px', fontWeight: '600',
            }
        }, count + ' obs'),
    ]);
};

/**
 * ActionsCell — multiple viewer link buttons per row (All DP tab).
 * props.value: JSON string of [{label, href, color}, ...]
 */
dagcomponentfuncs.ActionsCell = function (props) {
    var val = props.value;
    if (!val) return React.createElement('span', { style: { color: '#aaa' } }, '—');
    var actions;
    try { actions = JSON.parse(val); } catch (e) { return null; }
    if (!actions || !actions.length) return React.createElement('span', { style: { color: '#aaa' } }, '—');
    var children = actions.map(function (a, i) {
        return React.createElement('a', {
            key: i,
            href: a.href,
            style: {
                color: a.color || '#228be6',
                fontWeight: '600', fontSize: '12px', textDecoration: 'none',
                whiteSpace: 'nowrap',
            },
        }, a.label);
    });
    return React.createElement('div', {
        style: { display: 'flex', gap: '10px', alignItems: 'center', height: '100%' },
    }, children);
};

/**
 * AssocLink — "assoc." collection-filter link (All DP tab).
 * props.value: assoc_key string (master-obsnum-sub-scan), or null.
 * Clicking navigates to /?collection=<key>, filtering the table to associated dp.
 */
dagcomponentfuncs.AssocLink = function (props) {
    var key = props.value;
    if (!key) return React.createElement('span', { style: { color: '#aaa' } }, '');
    var params = new URLSearchParams(window.location.search || '');
    var isActive = params.get('collection') === key;
    if (isActive) {
        return React.createElement('a', {
            href: '/',
            style: { color: '#1971c2', fontSize: '11px', fontWeight: '600', textDecoration: 'none' },
        }, '\u2190 all');
    }
    params.set('collection', key);
    var href = '/?' + params.toString();
    return React.createElement('a', {
        href: href,
        style: {
            color: '#868e96', fontSize: '11px', fontWeight: '500', textDecoration: 'none',
            background: '#f1f3f5', borderRadius: '4px', padding: '2px 7px',
        },
    }, 'assoc.');
};

/**
 * SweepLink — legacy single-link renderer (kept for backward compat).
 * Uses props.data.sweep_href.
 */
dagcomponentfuncs.SweepLink = function (props) {
    var href = (props.data || {}).sweep_href;
    if (!href) return React.createElement('span', { style: { color: '#aaa' } }, '—');
    var isKids = href.indexOf('/kids-diag') === 0 || href.indexOf('/reduced-obs') === 0;
    return React.createElement('a', {
        href: href,
        style: {
            color: isKids ? '#7048e8' : '#228be6',
            fontWeight: '600', fontSize: '12px', textDecoration: 'none',
        },
    }, isKids ? 'Diagnostics \u2192' : 'SweepViewer \u2192');
};
