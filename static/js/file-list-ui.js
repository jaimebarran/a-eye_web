const VOLUME_LABELS = {
    vol_lens: 'lens',
    vol_globe: 'globe',
    vol_nerve: 'optic nerve',
    vol_int_fat: 'intraconal fat',
    vol_ext_fat: 'extraconal fat',
    vol_lat_mus: 'lateral rectus muscle',
    vol_med_mus: 'medial rectus muscle',
    vol_inf_mus: 'inferior rectus muscle',
    vol_sup_mus: 'superior rectus muscle',
    vol_total: 'Total segmented volume',
};

const AXIAL_LENGTH_LABELS = {
    axial_length_cornea: 'axial length (until cornea)',
};


/**
* Compute the dataset of the case's aggregate, mean and standard deviation for each biomarker
* @param {array} results - results of segmentation, with biomarkers object
* @param {string} side - "left" or "right" eye
* @param {array} biomarkers - biomarkers to compute
* @returns {object} - key (biomarker) -> {mean, std}
*/
/**
 * Adapted from (Online computation of mean and variance, accessed 08.07.2026)
 * URL: https://medium.com/@giacinti.2224996/online-computation-of-mean-and-variance-cd8d741c8e54
 */
function computeSelectionStats(results, side, biomarkers) {
    const statistics = {};
    for (const key of biomarkers) {
        let n = 0, mean = 0, m2 = 0;
        for (const result of results) {
            const value = result.biomarkers[side][key];
            if (value === null || value === undefined) continue;
            n++;
            const delta = value - mean;
            mean += delta / n;
            m2 += delta * (value - mean);
        }

        if (!n) {
            statistics[key] = { mean: null, std: null };
            continue;
        }

        statistics[key] = { mean, std: Math.sqrt(n > 1 ? m2 / (n - 1) : 0) };
    }
    return statistics;
}


/**
* Add a panzoom component a the image. 
* Enable to zoom on it with Shift + mouse Wheel.
* @param {Element} row - The container to search in : '.wheel-zoom img', to attach panzoom to
*/
/**
 * Adapted from (panzoom, accessed 04.07.2026)
 * URL: https://github.com/timmywil/panzoom
 * URL: https://timmywil.com/panzoom/demo/
 */
function addMouseWheelZoom(row) {
    row.querySelectorAll('.wheel-zoom img').forEach((image) => {
        const panzoom = Panzoom(image, { minScale: 1, maxScale: 5, panOnlyWhenZoomed: true, canvas: true });
        image.panzoom = panzoom;

        const parent = image.parentElement;
        parent.addEventListener('wheel', function (event) {
            if (!event.shiftKey) return;
            panzoom.zoomWithWheel(event);
        });
    });
}

/**
* Render the outliers detected message
* @param {object} sideData - The key-values object containing measurements and outliers
* @param {object} labelsMap - Mapping of biomarker key to its display label
*/
function renderOutlierAlert(sideData, labelsMap) {
    const femaleLabels = [];
    const maleLabels = [];
    for (const key in labelsMap) {
        if (sideData.outliers.F[key]) {
            femaleLabels.push(labelsMap[key]);
        }
        if (sideData.outliers.M[key]) {
            maleLabels.push(labelsMap[key]);
        }
    }
    
    if (!femaleLabels.length && !maleLabels.length) {
        return '';
    }

    const message = [];
    if (femaleLabels.length) {
        message.push(`<br> → if <strong>female</strong>: <strong>${femaleLabels.join(', ')}</strong>`);
    }
    if (maleLabels.length) {
        message.push(`<br> → if <strong>male</strong>: <strong>${maleLabels.join(', ')}</strong>`);
    }

    return `<div class="alert alert-warning mb-2" role="alert">
                <i class="fa-solid fa-triangle-exclamation"></i>
                Comparing the extracted results and the reference mean revealed outliers
                depending on the case's sex : ${message.join('')}
            </div>`;
}

/**
 * Render the metadata table
 * @param {string} fields - The key-values object containing the metadata
 */

function renderMetadataTable(fields) {
    // check that the fields dictionnary isn't empty
    if (!Object.keys(fields).length) {
        return `<p class="text-warning d-block"> Metadata not available...</p>`
    }
    let row = ``;
    for (const key in fields) {
        const value = fields?.[key];
        const cleanedValue = Array.isArray(value) ? value.join(', ') : String(value);

        row += `
            <tr>
                <td class="label-cell">${key}</td>
                <td class="value-cell">${cleanedValue}</td>
            </tr>`;
    }
    return `
    <table class="measure-table">
        <thead>
            <tr>
                <th>Fields</th>
                <th>Values</th>
            </tr>
        </thead>
        <tbody>
            ${row}
        </tbody>
    </table>`;
}


/*
*   To render the legends of ROI after segmentation
*/
function renderSegmentationLegend() {
    const legends = window.EYE_LABELS.slice(1).map(({ name, color }) =>
        `<div class="seg-legend-bloc">
            <span class="seg-legend-swatch" style="background: rgb(${color})"></span>
            <span>${name}</span>
        </div>`
    ).join('');
    return `<div class="seg-legend">
                <strong>Labels of segmentation</strong>${legends}
            </div>`;
}


/**
* Render the axial length measurements table
* @param {object} sideData - The key-values object containing the AL measurements
* @param {bool} displayReference - wether to display the reference too
**/
function renderAxialLengthTable(sideData, displayReference = false) {
    if (!Object.keys(sideData).length) {
        return `<p class="text-warning d-block"> No axial length data available...</p>`
    }
    let row = ``;
    const data = displayReference ? [['Axial length (until cornea)', `axial_length_cornea`, 'mm']] : [
            ['Axial length (until lens)', `axial_length`, 'mm'],
            ['Axial length (until cornea)', `axial_length_cornea`, 'mm'],
            ['Extra anterior', `extra_ant`, 'mm'],
        ];

    for (const [label, key, unit] of data) {
        const value = sideData?.[key];
        const cleanedValue = (value !== null && value !== undefined) ? `${value.toFixed(2)} ${unit}` : '—';
        let refTd = ``;
        let rowClass = ``;
        if (displayReference) {
            const refValue = sideData?.reference_mean?.[key];
            const refCleanedValue = (refValue !== null && refValue !== undefined)
                                     ? `${refValue.toFixed(2)} ${unit}` : '—';
            refTd = `<td class="value-cell">${refCleanedValue}</td>`;
            rowClass = sideData.outliers.F[key] || sideData.outliers.M[key] ? ` class="active-row"` : ``;
        }
        row += `
                <tr${rowClass}>
                    <td class="label-cell">${label}</td>
                    <td class="value-cell">${cleanedValue}</td>
                    ${refTd}
                </tr>`;
    }
    let refHeader = ``
    if (displayReference) {
        refHeader = `<th>Reference mean</th>`;
        return `
            <table class="measure-table-ref">
            <thead>
                <tr>
                    <th>Measures</th>
                    <th>Values</th>
                    ${refHeader}
                </tr>
            </thead>
            <tbody>
                ${row}
            </tbody>
        </table>`;
    }
    return `
            <table class="measure-table">
            <thead>
                <tr>
                    <th>Measures</th>
                    <th>Values</th>
                    ${refHeader}
                </tr>
            </thead>
            <tbody>
                ${row}
            </tbody>
        </table>`;
}


/**
* Render the volumetry measurements table
* @param {object} sideData - The key-values object containing the AL measurements
* @param {bool} displayReference - wether to display the reference too
*/
function renderVolumetryTable(sideData, displayReference = false) {

    if (!Object.keys(sideData).length) {
        return `<p class="text-warning d-block"> No volumetry data available...</p>`
    }

    let row = ``;
    const data = Object.entries(VOLUME_LABELS)

    for (const [key, label] of data) {
        const value = sideData?.[key];
        const cleanedValue = (value !== null && value !== undefined) ? `${value.toFixed(2)} mm³` : '—';
        let refTd = ``;
        let rowClass = ``;

        if (displayReference) {
            const refValue = sideData?.reference_mean?.[key];
            const refCleanedValue = (refValue !== null && refValue !== undefined) ? `${refValue.toFixed(2)} mm³` : '—';
            refTd = `<td class="value-cell">${refCleanedValue}</td>`;
            rowClass = sideData.outliers.F[key] || sideData.outliers.M[key] ? ` class="active-row"` : ``;
        }

        row += `
            <tr${rowClass}>
                <td class="label-cell">${label}</td>
                <td class="value-cell">${cleanedValue}</td>
                ${refTd}
            </tr>`;
    }
    let refHeader = ``;
    if (displayReference) {
        refHeader = `<th>Reference mean</th>`;
        return `
            <table class="measure-table-ref">
                <thead>
                    <tr>
                        <th>Region</th>
                        <th>Volume</th>
                        ${refHeader}
                    </tr>
                </thead>
                <tbody>
                    ${row}
                </tbody>
            </table>`;
    }

    return `
        <table class="measure-table">
            <thead>
                <tr>
                    <th>Region</th>
                    <th>Volume</th>
                    ${refHeader}
                </tr>
            </thead>
            <tbody>
                ${row}
            </tbody>
        </table>`;
}

/**
* Build the HTML content shown inside the segmentation dropdown row
* @param {object} result - the case result
* @param {number} idx - the row index, to build unique elements
*/
function renderSegmentationDropdownContent(result, idx) {
    return `
        <div class="d-flex align-items-center gap-2 mb-3">
                <h4 class="mb-0 fw-semibold" style="color: var(--stat-value-color)">Segmentation</h4>
                <hr class="flex-grow-1 my-0" style="border-top: 1px solid var( --summary-link-color);">
        </div>
        <div style="display:flex; width:100%; gap:8px; align-items:flex-start;">
            <div style="flex:1; min-width:0; display:flex; flex-direction:column; gap:4px;">
                <span class="text-muted small fw-semibold" style="margin-left:2px;">
                    Both eyes overlaid on original image
                </span>
                <canvas id="niivue-overlay-${idx}" style="display:block; width:100%; aspect-ratio:1/1;"></canvas>
            </div>
            <div id="info-panel-${idx}" style="flex:0 0 20%; overflow-y:auto;
                                                 border-left:2px solid var(--info-panel-border); 
                                                 border-right:2px solid var(--info-panel-border);
                                                 padding:0 8px;">
                ${renderSegmentationLegend()}
                <div style="margin-top:8px; font-size:0.78em;">
                    <strong style="margin-bottom:5px;">Metadata</strong>
                </div>
                <div>${renderMetadataTable(result.metadata)}</div>
            </div>
            <div style="flex:1; min-width:0; display:flex; flex-direction:column; gap:4px;">
                <span class="text-muted small fw-semibold" style="margin-left:2px;">
                    Both eyes only
                </span>
                <canvas id="niivue-eyes-${idx}" style="display:block; width:100%; aspect-ratio:1/1;"></canvas>
            </div>
        </div>`;
}

/**
* Build the HTML content shown inside the biomarkers dropdown row
* @param {object} results - containing per-eye biomarkers object with references
*/
function renderBiomarkersDropdownContent(results) {
    const title = `<div class="d-flex align-items-center gap-2 mb-3">
                        <h4 class="mb-0 fw-semibold" style="color: var(--stat-value-color)">Biomarkers</h4>
                        <hr class="flex-grow-1 my-0" style="border-top: 1px solid var( --summary-link-color);">
                </div>
                <p style="color: var(--muted-text); font-size: 0.85rem; margin-bottom: 1rem;">
                    This section present the extracted biomarkers : axial length and volumetry.
                </p>`;
    const details = `<details class="mb-3" style="font-size:0.82em;">
                        <summary style="cursor: pointer; font-weight: 600; list-style: none; display: inline-flex;
                                         align-items: center; gap: 6px; color:var(--summary-link-color); 
                                         line-height:1.5;">
                            <i class="fa-solid fa-circle-info"></i> How are biomarkers computed?
                        </summary>

                        <p class="mt-2 mb-1">
                            <strong>Axial length</strong><br>
                            The axial length is defined as the distance from the anterior surface of the cornea to 
                             the posterior pole of the ocular bulb. The base measurement spans from the posterior lens 
                            surface to the posterior globe surface; the corneal extension is added separately 
                            to obtain the full axial length. The method inputs both the automated 
                            segmented labels and T1w images. A ray is traced between the centroids of the lens
                             and globe segmentations and extended to the full volume bounding box. The anterior lens 
                            edge and posterior globe edge are identified as the extreme intersection points of that ray
                             with those structures. The corneal boundary is then estimated 
                             from the T1-weighted intensity gradient along the same ray. It corresponds to the first 
                             gradient peak after the first major drop in the region anterior to the lens.
                             Both eyes are extracted independently.
                        </p>
                        <p class="mb-1"><strong>Volumetry</strong><br>
                            The volume of each segmented structure is computed by counting
                             the number of voxels belonging to that label
                            and multiplying by the physical voxel volume (in mm³).
                        </p>
                        <p class="mt-1 mb-0 text-muted">
                            Reference: 
                                <em>
                                    <a href="https://doi.org/10.1371/journal.pone.0352257">A-eye paper </a>
                                </em>
                        </p>
                    </details>`;
    let columns = ``;
    const sides = [`left`, `right`];

    for (const side of sides) {
        const data = results?.[side];

        if (!data || data.error) {
            columns += `
                    <div class="col-12 col-md-6">
                        <p class="text-warning d-block">${data?.error || 'No data'}</p>
                    </div>`;
        } else {
            columns += `
                    <div class="col-12 col-md-6">
                        <div class="card h-100 shadow-sm border-0">
                            <div class="card-header d-flex align-items-center gap-2 justify-content-center" 
                            style="background:var(--card-bg);">
                                <i class="fa-solid fa-eye fa-fw text-success"></i>
                                <strong class="text-capitalize">${side} eye</strong>
                            </div>
                            <div class="card-body d-flex flex-column gap-4">
                                <div>
                                    <h5 style="color: var(--stat-value-color); text-align: center;">
                                        Volumetry
                                    </h5>
                                    ${renderVolumetryTable(data)}
                                </div>
                                <div>
                                    <h5 style="color: var(--stat-value-color); text-align: center;">
                                        Axial length
                                    </h5>
                                    ${data.axial_length_image ? `
                                        <img src="${data.axial_length_image}" class="img-fluid bg-black mb-2"
                                            style="border: 1px solid var(--header-divider); border-radius: 4px;"
                                            alt="Axial length ${side} eye">` : ''}
                                    ${renderAxialLengthTable(data)}
                                </div>
                            </div>
                        </div>
                    </div>`;
        }
    }

    return `<div>
    ${title}
    ${details}
    <div class="row g-3"> ${columns} </div>
</div>`;
}

/**
* Build the HTML content shown inside the statistical analysis dropdown row
* @param {object} results - containing per-eye biomarkers object with references
*/
function renderStatisticalDropdownContent(results) {
    const title = `<div class="d-flex align-items-center gap-2 mb-3">
                    <h4 class="mb-0 fw-semibold" style="color: var(--stat-value-color)">Statistics</h4>
                    <hr class="flex-grow-1 my-0" style="border-top: 1px solid var( --summary-link-color);">
            </div>
            <p style="color: var(--muted-text); font-size: 0.85rem; margin-bottom: 1rem;">
                This section compare the extracted biomarkers to the reference dataset
                 (composed of 1157 samples, processed with quadrant segmentation). The "Reference mean" value correspond
                 to both sex samples combined.
            </p>`;
    let columns = ``;
    const sides = [`left`, `right`];

    for (const side of sides) {
        const data = results?.[side];

        if (!data || data.error) {
            columns += `
                    <div class="col-12 col-md-6">
                        <p class="text-danger">${data?.error || 'No data'}</p>
                    </div>`;
        } else {
            columns += `
                    <div class="col-12 col-md-6">
                        <div class="card h-100 shadow-sm border-0">
                            <div class="card-header d-flex align-items-center gap-2 justify-content-center"
                                style="background:var(--card-bg);">
                                <i class="fa-solid fa-eye fa-fw text-success"></i>
                                <strong class="text-capitalize">${side} eye</strong>
                            </div>
                            <div class="card-body d-flex flex-column gap-4">

                                <div>
                                    <h5 style="color: var(--stat-value-color); text-align: center;">
                                        Volumetry
                                    </h5>
                                    ${renderOutlierAlert(data, VOLUME_LABELS)}
                                    ${renderVolumetryTable(data, true)}
                                </div>

                                ${data.vol_violin_image ? `
                                    <div>
                                        <div class='wheel-zoom'>
                                            <img src="${data.vol_violin_image}" class="img-fluid bg-black"
                                            style="border: 1px solid var(--header-divider); border-radius: 4px;"
                                            alt="Volumetry violin plot ${side} eye">
                                        </div>
                                        <p class="text-center mt-2 mb-0">
                                            <span class="zoom-hint">
                                                <kbd>Shift</kbd> + <kbd>mouse wheel</kbd> to zoom &middot;
                                                <kbd>Esc</kbd> to reset view
                                            </span>
                                        </p>
                                    </div>` : ''}
                                <div>
                                    <h5 style="color: var(--stat-value-color); text-align: center;">
                                        Axial length
                                    </h5>
                                    ${renderOutlierAlert(data, AXIAL_LENGTH_LABELS)}
                                    ${renderAxialLengthTable(data, true)}
                                </div>

                                ${data.al_violin_image ? `
                                    <div>
                                        <img src="${data.al_violin_image}" class="img-fluid bg-black"
                                        style="border: 1px solid var(--header-divider); border-radius: 4px;"
                                        alt="Axial length violin plot ${side} eye">
                                    </div>` : ''}

                            </div>
                        </div>
                    </div>`;
        }
    }
    return `<div class="row g-3">
            ${title}
            ${columns}
            </div>`;
}

/**
* Render a comparison table (mean & standard deviation) between the aggregate dataset of selection cases
* and the reference dataset
* @param {object} labelsMap - biomarker key mapped to it's display label
* @param {object} selectionStats - key -> {mean, std} of the selected cases
* @param {object} referenceMean - key -> mean of the reference dataset
* @param {object} referenceStd - key -> standard deviation of the referecence dataset
* @param {number} caseCount - number of selected cases aggregated
* @param {number} referenceCount - number of sample in the reference dataset
*/
function renderComparisonTable(labelsMap, selectionStats, referenceMean, referenceStd, caseCount, referenceCount) {
    const unit = labelsMap === VOLUME_LABELS ? `mm³` : `mm`;
    let row = ``;
    for (const key in labelsMap) {
        const selection = selectionStats[key];
        const selMean = selection?.mean != null ? `${selection.mean.toFixed(2)} ${unit}` : '—';
        const selStd = selection?.std != null ? `${selection.std.toFixed(2)} ${unit}` : '—';
        const refMeanValue = referenceMean?.[key];
        const refStdValue = referenceStd?.[key];
        const refMean = refMeanValue != null ? `${refMeanValue.toFixed(2)} ${unit}` : '—';
        const refStd = refStdValue != null ? `${refStdValue.toFixed(2)} ${unit}` : '—';

        row += `
            <tr>
                <td class="label-cell">${labelsMap[key]}</td>
                <td class="value-cell">${selMean}</td>
                <td class="value-cell">${selStd}</td>
                <td class="value-cell ref-cell">${refMean}</td>
                <td class="value-cell ref-cell">${refStd}</td>
            </tr>`;
    }
    return `
        <table class="measure-table-dataset">
            <thead>
                <tr>
                    <th class="label-cell" rowspan="2">Measures</th>
                    <th colspan="2">Selected cases (n=${caseCount})</th>
                    <th colspan="2" class="ref-cell">Reference dataset (n=${referenceCount})</th>
                </tr>
                <tr>
                    <th>Mean</th>
                    <th>SD</th>
                    <th class="ref-cell">Mean</th>
                    <th class="ref-cell">SD</th>
                </tr>
            </thead>
            <tbody>
                ${row}
            </tbody>
        </table>`;
}

/**
* Build the HTML content shown inside the accordion comparing the aggregate case's result and reference dataset
* @param {array} results - segmentation results, each optionally holding a `biomarkers` object
*/
function renderComparisonAccordion(results) {
    const sides = [`left`, `right`];
    let columns = ``;

    for (const side of sides) {
        let reference = null;
        for (const result of results) {
            const data = result.biomarkers[side];
            // only using the first case reference
            if (data && !data.error && data.reference_mean) {
                reference = data;
                break;
            }
        }

        if (!reference) {
            columns += `
                <div class="col-12 col-md-6">
                    <p class="text-warning d-block">No reference data available for the ${side} eye</p>
                </div>`;
        } else {
            const volStats = computeSelectionStats(results, side, Object.keys(VOLUME_LABELS));
            const alStats = computeSelectionStats(results, side, Object.keys(AXIAL_LENGTH_LABELS));

            columns += `
                <div class="col-12 col-md-6">
                    <div class="card h-100 shadow-sm border-0">
                        <div class="card-header d-flex align-items-center gap-2 justify-content-center"
                            style="background:var(--card-bg);">
                            <i class="fa-solid fa-eye fa-fw text-success"></i>
                            <strong class="text-capitalize">${side} eye</strong>
                        </div>
                        <div class="card-body d-flex flex-column gap-4">
                            <div>
                                <h5 style="color: var(--stat-value-color); text-align: center;">
                                    Volumetry
                                </h5>
                                ${renderComparisonTable(VOLUME_LABELS, volStats, reference.reference_mean,
                                                         reference.reference_std, results.length,
                                                         reference.reference_count.volumetry)}
                            </div>
                            <div>
                                <h5 style="color: var(--stat-value-color); text-align: center;">
                                    Axial length
                                </h5>
                                ${renderComparisonTable(AXIAL_LENGTH_LABELS, alStats, reference.reference_mean,
                                                         reference.reference_std, results.length,
                                                         reference.reference_count.axial_length)}
                            </div>
                        </div>
                    </div>
                </div>`;
        }
    }

    return `<p style="color: var(--muted-text); font-size: 0.85rem; margin-bottom: 1rem;">
                This section present a biomarkers comparison between a dataset of the aggregated values of
                 the case's processed and the reference dataset.
            </p>
            <div class="row g-3">${columns}</div>`;
}

// Appends sub-rows for each case in a selected file
function addLabelRow(ulElement, labels) {
    labels.forEach(label => {
        const row = document.createElement('li');
        row.className = 'list-group-item fl-case-label';

        const qualifier = document.createElement('span');
        qualifier.style.color = 'var(--summary-link-color)';
        qualifier.style.fontWeight ='bold';
        qualifier.textContent = 'CASE · ';
        row.appendChild(qualifier);
        row.appendChild(document.createTextNode(label));

        ulElement.appendChild(row);
    });
}

/*
*   Build a file list with an optional remove button and dropdowns buttons
*/
function buildFileList(ulElement, files, onRemove = null, dropdowns = [], caseInfoMap = null, headerBadge = null) {
    ulElement.innerHTML = '';

    const header = document.createElement('li');
    header.className = 'list-group-item d-flex align-items-center gap-2';
    header.style.backgroundColor = 'var(--progress-bg)';
    header.style.color = 'var(--text-color)';

    const headerFilename = document.createElement('span');
    headerFilename.style.fontWeight = 'bold';
    headerFilename.textContent = 'Selection';
    header.appendChild(headerFilename);

    if (headerBadge) {
        const badge = document.createElement('span');
        badge.innerHTML = headerBadge;
        header.appendChild(badge);
    }

    ulElement.appendChild(header);

    files.forEach((file, index) => {
        const fileRow = document.createElement('li');
        fileRow.className = 'list-group-item fl-item-row d-flex justify-content-between align-items-center';

        const nameSpan = document.createElement('span');
        const qualifier = document.createElement('span');
        qualifier.style.color = file instanceof File ? 'var(--link-color)' : 'var(--summary-link-color)';
        qualifier.textContent = file instanceof File ? 'FILE · ' : 'CASE · ';
        qualifier.style.fontWeight = 'bold';
        nameSpan.appendChild(qualifier);
        nameSpan.appendChild(document.createTextNode(file.name));

        fileRow.appendChild(nameSpan);

        if (onRemove) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'btn btn-danger btn-sm';
            btn.textContent = 'remove';
            btn.onclick = () => onRemove(index);
            fileRow.appendChild(btn);
        }

        if (dropdowns.length > 0) {
            const btnGroup = document.createElement('div');
            btnGroup.className = 'd-flex gap-2';
            fileRow.appendChild(btnGroup);
            ulElement.appendChild(fileRow);

            dropdowns.forEach(({ label, onOpen, onClose }) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'btn btn-primary btn-sm';
                btn.textContent = label;
                btnGroup.appendChild(btn);

                const dropdownRow = document.createElement('li');
                dropdownRow.className = 'list-group-item';
                dropdownRow.style.display = 'none';
                ulElement.appendChild(dropdownRow);

                btn.onclick = async () => {
                    const collapsed = dropdownRow.style.display === 'none';
                    dropdownRow.style.display = collapsed ? 'block' : 'none';
                    btn.textContent = collapsed ? 'hide' : label;
                    if (collapsed) {
                        await onOpen(file, index, dropdownRow);
                    } else if (onClose) {
                        onClose(file, index, dropdownRow);
                    }
                };
            });
        } else {
            ulElement.appendChild(fileRow);
        }

        const info = caseInfoMap?.get(file);
        if (info?.labels) {
            addLabelRow(ulElement, info.labels);
        }
    });
}