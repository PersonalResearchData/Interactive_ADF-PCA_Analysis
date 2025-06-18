let uploadedFiles = [];
let processedData = null;

// File upload handling
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const processBtn = document.getElementById('processBtn');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    uploadedFiles = Array.from(files).filter(file => file.name.endsWith('.xyz'));
    displayFileList();
    processBtn.disabled = uploadedFiles.length === 0;
}

function displayFileList() {
    fileList.innerHTML = '';
    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span class="file-name">${file.name}</span>
            <span class="file-size">${(file.size / 1024).toFixed(1)} KB</span>
        `;
        fileList.appendChild(fileItem);
    });
}

// XYZ file parser
function parseXYZFile(content) {
    const lines = content.trim().split('\n');
    if (lines.length < 2) throw new Error('Invalid XYZ file format');

    const numParticles = parseInt(lines[0]);
    if (isNaN(numParticles)) throw new Error('Invalid particle count');

    const boxLine = lines[1].split(/\s+/);
    const boxIndex = boxLine.indexOf('box');
    if (boxIndex === -1 || boxLine.length < boxIndex + 7) {
        throw new Error('Invalid box information');
    }

    const [xlo, xhi, ylo, yhi, zlo, zhi] = boxLine.slice(boxIndex + 1, boxIndex + 7).map(parseFloat);
    const boxSize = [xhi - xlo, yhi - ylo, zhi - zlo];

    const positions = [];
    for (let i = 2; i < 2 + numParticles && i < lines.length; i++) {
        const parts = lines[i].split(/\s+/);
        if (parts.length >= 4) {
            positions.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
        }
    }

    return { positions, boxSize };
}

// ADF calculation (from original ADF program)
function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function computeADF(positions, boxSize, rMax, numBins) {
    const numParticles = positions.length;
    const thetaBin = Array.from({ length: numBins + 1 }, (_, i) => (i * 180) / numBins);
    const dtheta = thetaBin[1] - thetaBin[0];
    const adf = new Array(numBins).fill(0);

    for (let i = 0; i < numParticles; i++) {
        const delta = positions.map(pos => [
            pos[0] - positions[i][0],
            pos[1] - positions[i][1],
            pos[2] - positions[i][2]
        ]);
        
        // Apply periodic boundary conditions
        delta.forEach((d, idx) => {
            for (let j = 0; j < 3; j++) {
                d[j] = d[j] - boxSize[j] * Math.round(d[j] / boxSize[j]);
            }
        });

        const distances = delta.map(d => Math.sqrt(dot(d, d)));
        const neighbors = distances
            .map((d, idx) => (d > 0 && d < rMax ? idx : -1))
            .filter(idx => idx !== -1);

        for (let jIdx = 0; jIdx < neighbors.length; jIdx++) {
            for (let kIdx = jIdx + 1; kIdx < neighbors.length; kIdx++) {
                const j = neighbors[jIdx];
                const k = neighbors[kIdx];
                const vecIJ = delta[j];
                const vecIK = delta[k];
                const normIJ = Math.sqrt(dot(vecIJ, vecIJ));
                const normIK = Math.sqrt(dot(vecIK, vecIK));
                if (normIJ === 0 || normIK === 0) continue;

                let cosTheta = dot(vecIJ, vecIK) / (normIJ * normIK);
                cosTheta = Math.min(1.0, Math.max(-1.0, cosTheta));
                const theta = (Math.acos(cosTheta) * 180) / Math.PI;
                const binIndex = Math.floor(theta / dtheta);
                if (binIndex >= 0 && binIndex < numBins) {
                    adf[binIndex] += 2;
                }
            }
        }
    }

    const totalCounts = adf.reduce((sum, val) => sum + val, 0);
    if (totalCounts === 0) {
        console.warn("No angles were counted within r_max.");
        return { theta: thetaBin.slice(0, -1), adf };
    }

    const dthetaRad = (dtheta * Math.PI) / 180;
    const normalizedADF = adf.map(count => count / (totalCounts * dthetaRad));
    return { theta: thetaBin.slice(0, -1), adf: normalizedADF };
}

// Matrix operations for PCA
function transpose(matrix) {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

function multiplyMatrices(a, b) {
    const result = Array(a.length).fill().map(() => Array(b[0].length).fill(0));
    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < b[0].length; j++) {
            for (let k = 0; k < a[0].length; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

// SVD-based PCA implementation
function performPCA(data, nComponents) {
    const n = data.length;
    const m = data[0].length;
    nComponents = Math.min(nComponents, n, m);

    // Center the data
    const means = new Array(m).fill(0);
    for (let j = 0; j < m; j++) {
        for (let i = 0; i < n; i++) {
            means[j] += data[i][j];
        }
        means[j] /= n;
    }

    const centeredData = data.map(row => row.map((val, j) => val - means[j]));

    // Compute covariance matrix
    const X = centeredData;
    const Xt = transpose(X);
    const cov = multiplyMatrices(Xt, X);
    
    // Scale by (n-1) for unbiased covariance
    for (let i = 0; i < cov.length; i++) {
        for (let j = 0; j < cov[i].length; j++) {
            cov[i][j] /= (n - 1);
        }
    }

    // Power iteration method for eigenvalue decomposition
    const eigenPairs = [];
    const covCopy = cov.map(row => [...row]); // Copy for modification

    for (let comp = 0; comp < Math.min(nComponents, 10); comp++) {
        // Power iteration to find dominant eigenvector
        let v = new Array(m).fill(0).map(() => Math.random() - 0.5);
        let lambda = 0;

        for (let iter = 0; iter < 100; iter++) {
            // v = A * v
            const newV = new Array(m).fill(0);
            for (let i = 0; i < m; i++) {
                for (let j = 0; j < m; j++) {
                    newV[i] += covCopy[i][j] * v[j];
                }
            }

            // Compute eigenvalue (Rayleigh quotient)
            const vDotNewV = v.reduce((sum, val, i) => sum + val * newV[i], 0);
            const vDotV = v.reduce((sum, val) => sum + val * val, 0);
            lambda = vDotNewV / vDotV;

            // Normalize
            const norm = Math.sqrt(newV.reduce((sum, val) => sum + val * val, 0));
            if (norm > 1e-10) {
                v = newV.map(val => val / norm);
            }
        }

        eigenPairs.push({ value: Math.abs(lambda), vector: v });

        // Deflate matrix for next eigenvector
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < m; j++) {
                covCopy[i][j] -= lambda * v[i] * v[j];
            }
        }
    }

    // Sort by eigenvalue (descending)
    eigenPairs.sort((a, b) => b.value - a.value);

    // Project data onto principal components
    const pcaResult = centeredData.map(row => {
        const projected = [];
        for (let comp = 0; comp < nComponents; comp++) {
            const pc = eigenPairs[comp].vector;
            const projection = row.reduce((sum, val, i) => sum + val * pc[i], 0);
            projected.push(projection);
        }
        return projected;
    });

    // Calculate explained variance
    const totalVariance = eigenPairs.reduce((sum, pair) => sum + pair.value, 0);
    const explainedVariance = eigenPairs.slice(0, nComponents).map(pair => 
        pair.value / totalVariance
    );

    const cumulativeVariance = [];
    let cumSum = 0;
    for (let i = 0; i < explainedVariance.length; i++) {
        cumSum += explainedVariance[i];
        cumulativeVariance.push(cumSum);
    }

    return {
        pcaResult,
        explainedVariance,
        cumulativeVariance,
        eigenvalues: eigenPairs.slice(0, nComponents).map(p => p.value),
        eigenvectors: eigenPairs.slice(0, nComponents).map(p => p.vector)
    };
}

// Timestep extraction and label generation
function extractTimestep(filename) {
    const match = filename.match(/output_timestep_(\d+)(?:_.*)?\.xyz$/);
    return match ? parseInt(match[1]) : null;
}

function generateLabel(timestep, filename, timestepSize = 1.0) {
    if (timestep === null) return filename;
    
    const timePs = timestep * timestepSize / 1000; // Convert to ps
    
    if (timestep === 0) return "100 K, 0 fs";
    
    if (timePs < 1.0) {
        const timeFs = timestep * timestepSize;
        return `40 K, ${timeFs.toFixed(0)} fs`;
    } else if (timePs < 1000) {
        return `40 K, ${timePs.toFixed(2)} ps`;
    } else {
        const timeNs = timePs / 1000;
        return `40 K, ${timeNs.toFixed(2)} ns`;
    }
}

// Time averaging function
function performTimeAveraging(fileData, adfs, labels, timestepSize, averagingWindow) {
    if (fileData.length === 0) return { fileData, adfs, labels };

    // Convert averaging window from ps to timesteps
    const windowSteps = Math.round(averagingWindow / timestepSize * 1000);
    
    // Group files by time windows
    const timeGroups = new Map();
    
    fileData.forEach((item, index) => {
        const timestep = item.timestep || 0;
        const windowIndex = Math.floor(timestep / windowSteps);
        
        if (!timeGroups.has(windowIndex)) {
            timeGroups.set(windowIndex, []);
        }
        timeGroups.get(windowIndex).push({ index, timestep, item });
    });

    const averagedFileData = [];
    const averagedAdfs = [];
    const averagedLabels = [];

    // Average ADFs within each time window
    for (const [windowIndex, group] of timeGroups.entries()) {
        if (group.length === 0) continue;

        // Calculate average ADF for this window
        const avgAdf = new Array(adfs[0].length).fill(0);
        let count = 0;

        group.forEach(({ index }) => {
            for (let i = 0; i < adfs[index].length; i++) {
                avgAdf[i] += adfs[index][i];
            }
            count++;
        });

        // Normalize by count
        for (let i = 0; i < avgAdf.length; i++) {
            avgAdf[i] /= count;
        }

        // Calculate representative timestep (center of window)
        const centerTimestep = (windowIndex + 0.5) * windowSteps;
        const timePs = centerTimestep * timestepSize / 1000;

        averagedFileData.push({
            timestep: centerTimestep,
            windowIndex,
            count,
            timeRange: group.map(g => g.timestep)
        });
        averagedAdfs.push(avgAdf);
        
        // Create label for averaged data
        const label = `40 K, ${timePs.toFixed(1)} ps (avg of ${count} files)`;
        averagedLabels.push(label);
    }

    return {
        fileData: averagedFileData,
        adfs: averagedAdfs,
        labels: averagedLabels
    };
}

// Main processing function
async function processFiles() {
    if (uploadedFiles.length === 0) return;

    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    try {
        const rMax = parseFloat(document.getElementById('rMax').value);
        const numBins = parseInt(document.getElementById('numBins').value);
        const nComponents = parseInt(document.getElementById('pcComponents').value);
        const timestepSize = parseFloat(document.getElementById('timestepSize').value);
        const averagingWindow = parseFloat(document.getElementById('averagingWindow').value);
        const enableAveraging = document.getElementById('enableAveraging').checked;

        const fileData = [];
        const adfs = [];
        const labels = [];
        let thetaValues = null; // Store theta values from first file

        // Process each file
        for (const file of uploadedFiles) {
            const content = await file.text();
            const { positions, boxSize } = parseXYZFile(content);
            const { theta, adf } = computeADF(positions, boxSize, rMax, numBins);
            
            // Store theta values from first file (all files should have same theta values)
            if (thetaValues === null) {
                thetaValues = theta;
            }
            
            const timestep = extractTimestep(file.name);
            const label = generateLabel(timestep, file.name, timestepSize);
            
            fileData.push({ file, timestep, positions, boxSize });
            adfs.push(adf);
            labels.push(label);
        }

        // Sort by timestep before processing
        const sortedIndices = fileData
            .map((item, index) => ({ index, timestep: item.timestep }))
            .sort((a, b) => (a.timestep || Infinity) - (b.timestep || Infinity))
            .map(item => item.index);

        let finalFileData = sortedIndices.map(i => fileData[i]);
        let finalAdfs = sortedIndices.map(i => adfs[i]);
        let finalLabels = sortedIndices.map(i => labels[i]);

        // Apply time averaging if enabled
        if (enableAveraging) {
            const averaged = performTimeAveraging(finalFileData, finalAdfs, finalLabels, timestepSize, averagingWindow);
            finalFileData = averaged.fileData;
            finalAdfs = averaged.adfs;
            finalLabels = averaged.labels;
        }

        // Standardize data for PCA
        const means = new Array(finalAdfs[0].length).fill(0);
        const stds = new Array(finalAdfs[0].length).fill(0);

        for (let j = 0; j < finalAdfs[0].length; j++) {
            for (let i = 0; i < finalAdfs.length; i++) {
                means[j] += finalAdfs[i][j];
            }
            means[j] /= finalAdfs.length;
        }

        for (let j = 0; j < finalAdfs[0].length; j++) {
            for (let i = 0; i < finalAdfs.length; i++) {
                stds[j] += Math.pow(finalAdfs[i][j] - means[j], 2);
            }
            stds[j] = Math.sqrt(stds[j] / (finalAdfs.length - 1));
        }

        const standardizedAdfs = finalAdfs.map(adf => 
            adf.map((val, j) => (val - means[j]) / (stds[j] || 1))
        );

        // Perform PCA
        const pcaResults = performPCA(standardizedAdfs, nComponents);

        processedData = {
            theta: thetaValues,
            adfs: finalAdfs,
            labels: finalLabels,
            pca: pcaResults,
            numFiles: uploadedFiles.length,
            numBins: numBins,
            processedFiles: finalAdfs.length,
            averagingEnabled: enableAveraging,
            averagingWindow: averagingWindow,
            timestepSize: timestepSize
        };

        displayResults();

    } catch (error) {
        console.error('Processing error:', error);
        alert('Error processing files: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function displayResults() {
    if (!processedData) return;

    document.getElementById('results').style.display = 'block';

    // Initialize section toggles after results are displayed
    setTimeout(() => {
        initializeSectionToggles();
    }, 100);

    // Display statistics
    displayStats();

    // Plot ADFs
    plotADFs();

    // Plot PCA results
    plotPCA();

    // Plot contributions
    plotContributions();

    // Plot cumulative variance
    plotCumulative();
}

function displayStats() {
    const statsGrid = document.getElementById('statsGrid');
    
    let statsHTML = `
        <div class="stat-card">
            <h3>${processedData.numFiles}</h3>
            <p>Original Files</p>
        </div>
        <div class="stat-card">
            <h3>${processedData.processedFiles}</h3>
            <p>Processed Points</p>
        </div>
        <div class="stat-card">
            <h3>${processedData.numBins}</h3>
            <p>ADF Bins</p>
        </div>
        <div class="stat-card">
            <h3>${(processedData.pca.explainedVariance[0] * 100).toFixed(1)}%</h3>
            <p>PC1 Variance</p>
        </div>
    `;

    if (processedData.averagingEnabled) {
        statsHTML += `
            <div class="stat-card">
                <h3>${processedData.averagingWindow}</h3>
                <p>Averaging Window (ps)</p>
            </div>
            <div class="stat-card">
                <h3>${processedData.timestepSize}</h3>
                <p>Timestep Size (ps)</p>
            </div>
        `;
    }

    statsGrid.innerHTML = statsHTML;
}

function plotADFs() {
    const traces = processedData.adfs.map((adf, i) => ({
        x: processedData.theta,
        y: adf,
        type: 'scatter',
        mode: 'lines',
        name: processedData.labels[i],
        line: { width: 2 }
    }));

    const layout = {
        xaxis: { title: 'Angle (degrees)' },
        yaxis: { title: 'Density (1/radian)' },
        showlegend: true,
        legend: { x: 1.02, y: 1 },
        margin: { r: 150 }
    };

    Plotly.newPlot('adfPlot', traces, layout, { responsive: true });
}

function plotPCA() {
    // Extract timestep values for color mapping
    let colorValues = [];
    let colorBarTitle = 'File Index';
    
    if (processedData.averagingEnabled) {
        // For averaged data, use the center timestep of each window
        colorValues = processedData.labels.map((label, index) => {
            // Extract time from label like "40 K, 10.5 ps (avg of 3 files)"
            const timeMatch = label.match(/(\d+(?:\.\d+)?)\s*(fs|ps|ns)/);
            if (timeMatch) {
                let timeValue = parseFloat(timeMatch[1]);
                const unit = timeMatch[2];
                
                // Convert everything to ps for consistent scaling
                if (unit === 'fs') timeValue /= 1000;
                else if (unit === 'ns') timeValue *= 1000;
                
                return timeValue;
            }
            return index; // Fallback to index
        });
        colorBarTitle = 'Time (ps)';
    } else {
        // For individual files, extract timesteps from original file data
        colorValues = processedData.labels.map((label, index) => {
            // Try to extract timestep from the original file processing
            // Look for patterns like "100 K, 0 fs" or "40 K, 1.50 ps"
            const timeMatch = label.match(/(\d+(?:\.\d+)?)\s*(fs|ps|ns)/);
            if (timeMatch) {
                let timeValue = parseFloat(timeMatch[1]);
                const unit = timeMatch[2];
                
                // Convert everything to ps for consistent scaling
                if (unit === 'fs') timeValue /= 1000;
                else if (unit === 'ns') timeValue *= 1000;
                
                return timeValue;
            }
            return index; // Fallback to index
        });
        
        // Determine appropriate unit based on the range of values
        const maxTime = Math.max(...colorValues);
        if (maxTime < 1) {
            colorBarTitle = 'Time (fs)';
            colorValues = colorValues.map(t => t * 1000); // Convert back to fs
        } else if (maxTime < 1000) {
            colorBarTitle = 'Time (ps)';
        } else {
            colorBarTitle = 'Time (ns)';
            colorValues = colorValues.map(t => t / 1000); // Convert to ns
        }
    }

    const trace = {
        x: processedData.pca.pcaResult.map(point => point[0]),
        y: processedData.pca.pcaResult.map(point => point[1]),
        mode: 'markers+text',
        type: 'scatter',
        text: processedData.labels.map((label, i) => `${i+1}`), // Show point numbers
        textposition: 'top center',
        textfont: {
            size: 10,
            color: 'white'
        },
        marker: {
            size: 16,
            color: colorValues,
            colorscale: [
                [0, '#440154'],    // Dark purple (early times)
                [0.2, '#31688e'], // Dark blue
                [0.4, '#35b779'], // Green
                [0.6, '#fde725'], // Yellow
                [0.8, '#fd9731'], // Orange
                [1, '#cc4778']    // Pink-red (late times)
            ],
            colorbar: { 
                title: {
                    text: colorBarTitle,
                    font: { size: 14 }
                },
                thickness: 15,
                len: 0.7
            },
            line: {
                width: 2,
                color: 'white'
            }
        },
        hovertemplate: '<b>%{customdata}</b><br>' +
                      'PC1: %{x:.3f}<br>' +
                      'PC2: %{y:.3f}<br>' +
                      colorBarTitle + ': %{marker.color:.2f}<br>' +
                      '<b>Click to view ADF</b><br>' +
                      '<extra></extra>',
        customdata: processedData.labels
    };

    const layout = {
        xaxis: { 
            title: `PC1 (${(processedData.pca.explainedVariance[0] * 100).toFixed(2)}%)`,
            gridcolor: '#f0f0f0'
        },
        yaxis: { 
            title: `PC2 (${(processedData.pca.explainedVariance[1] * 100).toFixed(2)}%)`,
            gridcolor: '#f0f0f0'
        },
        hovermode: 'closest',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Arial, sans-serif' },
        title: {
            text: 'Click on any point to view its ADF',
            font: { size: 14, color: '#666' },
            x: 0.5
        }
    };

    const config = { 
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d']
    };

    Plotly.newPlot('pcaPlot', [trace], layout, config);

    // Add click event listener
    document.getElementById('pcaPlot').on('plotly_click', function(data) {
        if (data.points && data.points.length > 0) {
            const pointIndex = data.points[0].pointIndex;
            displaySelectedADF(pointIndex);
        }
    });
}

function displaySelectedADF(selectedIndex) {
    if (!processedData || selectedIndex >= processedData.adfs.length) return;

    // Show the selected ADF section if it's hidden
    const selectedAdfSection = document.getElementById('selected-adf-section');
    const selectedAdfToggle = document.querySelector('[data-section="selected-adf"]');
    
    if (!selectedAdfSection.classList.contains('visible')) {
        selectedAdfToggle.classList.add('active');
        selectedAdfToggle.classList.remove('inactive');
        selectedAdfSection.classList.add('visible');
        selectedAdfSection.style.display = 'block';
    }

    const selectedAdf = processedData.adfs[selectedIndex];
    const selectedLabel = processedData.labels[selectedIndex];

    // Update title
    document.getElementById('selectedAdfTitle').textContent = 
        `🔍 Selected ADF: ${selectedLabel}`;

    // Create trace for selected ADF
    const selectedTrace = {
        x: processedData.theta,
        y: selectedAdf,
        type: 'scatter',
        mode: 'lines',
        name: selectedLabel,
        line: { 
            width: 3, 
            color: '#e74c3c' 
        }
    };

    // Create comparison trace (all other ADFs in light gray)
    const allOtherTraces = processedData.adfs.map((adf, index) => {
        if (index === selectedIndex) return null;
        return {
            x: processedData.theta,
            y: adf,
            type: 'scatter',
            mode: 'lines',
            name: processedData.labels[index],
            line: { 
                width: 1, 
                color: 'rgba(150, 150, 150, 0.3)' 
            },
            showlegend: false,
            hovertemplate: `<b>${processedData.labels[index]}</b><br>θ: %{x:.1f}°<br>ADF: %{y:.3f}<extra></extra>`
        };
    }).filter(trace => trace !== null);

    const traces = [...allOtherTraces, selectedTrace];

    const layout = {
        xaxis: { 
            title: 'Angle (degrees)',
            gridcolor: '#f0f0f0'
        },
        yaxis: { 
            title: 'Density (1/radian)',
            gridcolor: '#f0f0f0'
        },
        showlegend: true,
        legend: { 
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.2,
            bgcolor: 'rgba(255,255,255,0.8)'
        },
        margin: { 
            l: 60, 
            r: 60, 
            t: 40, 
            b: 80 
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        hovermode: 'x unified'
    };

    const config = { 
        responsive: true,
        displayModeBar: true 
    };

    Plotly.newPlot('selectedAdfPlot', traces, layout, config);

    // Scroll to the selected ADF plot
    document.getElementById('selectedAdfPlot').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
    });
}

function plotContributions() {
    const nComponents = Math.min(processedData.pca.explainedVariance.length, 5);
    const components = Array.from({length: nComponents}, (_, i) => `PC${i+1}`);
    
    const trace = {
        x: components,
        y: processedData.pca.explainedVariance.slice(0, nComponents),
        type: 'bar',
        marker: {
            color: components.map((_, i) => `hsl(${240 + i * 30}, 70%, 60%)`)
        },
        text: processedData.pca.explainedVariance.slice(0, nComponents).map(v => `${(v * 100).toFixed(2)}%`),
        textposition: 'auto'
    };

    const layout = {
        xaxis: { title: 'Principal Component' },
        yaxis: { title: 'Explained Variance Ratio' },
        title: `Eigenvalues: ${processedData.pca.eigenvalues.slice(0, nComponents).map(v => v.toFixed(3)).join(', ')}`
    };

    Plotly.newPlot('contributionPlot', [trace], layout, { responsive: true });
}

function plotCumulative() {
    const nComponents = Math.min(processedData.pca.cumulativeVariance.length, 5);
    const components = Array.from({length: nComponents}, (_, i) => `PC${i+1}`);
    
    const trace = {
        x: components,
        y: processedData.pca.cumulativeVariance.slice(0, nComponents),
        type: 'scatter',
        mode: 'lines+markers',
        marker: { color: '#2ecc71', size: 10 },
        line: { color: '#2ecc71', width: 3 },
        text: processedData.pca.cumulativeVariance.slice(0, nComponents).map(v => `${(v * 100).toFixed(2)}%`),
        textposition: 'top center'
    };

    const layout = {
        xaxis: { title: 'Principal Component' },
        yaxis: { title: 'Cumulative Variance Ratio', range: [0, 1.1] }
    };

    Plotly.newPlot('cumulativePlot', [trace], layout, { responsive: true });
}

// Section toggle functionality
function initializeSectionToggles() {
    const toggleButtons = document.querySelectorAll('.section-toggle');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const sectionName = this.dataset.section;
            const section = document.getElementById(`${sectionName}-section`);
            
            if (!section) return;
            
            // Toggle button state
            if (this.classList.contains('active')) {
                this.classList.remove('active');
                this.classList.add('inactive');
                
                // Hide section
                section.classList.remove('visible');
                section.style.display = 'none';
            } else {
                this.classList.remove('inactive');
                this.classList.add('active');
                
                // Show section
                section.style.display = 'block';
                section.classList.add('visible');
                
                // Re-render the plot if it's being shown
                setTimeout(() => {
                    if (processedData) {
                        switch(sectionName) {
                            case 'adf':
                                plotADFs();
                                break;
                            case 'pca':
                                plotPCA();
                                break;
                            case 'contribution':
                                plotContributions();
                                break;
                            case 'cumulative':
                                plotCumulative();
                                break;
                        }
                    }
                }, 100);
            }
        });
    });
}

// Initialize toggles when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Will be initialized when displayResults() is called
});
