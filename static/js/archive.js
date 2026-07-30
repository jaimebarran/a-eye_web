let archiveWorker = null;

/**
* List the files contained in a .zip/.7z archive, via a web worker
* @param {File} file - the archive file to inspect
* @returns {Promise<string[]>} - resolves with the list of paths found in the archive
*/
/**
 * Adapted from (Promise practice of web worker, accessed 16.06.2026)
 * URL: https://byte-explorer.medium.com/promise-practice-of-web-worker-8c713d1c0754
 */
function listArchiveEntries(file) {
    return new Promise((resolve, reject) => {
        archiveWorker ??= new Worker('/static/js/archive-worker.js');
        let id = crypto.randomUUID();

        const onMessage = (event) => {
            if (event.data.id !== id) {
                return;
            }

            archiveWorker.removeEventListener('message', onMessage);

            if (event.data.error) {
                reject(event.data.error);
            } else {
                resolve(event.data.paths);
            }
        };

        archiveWorker.addEventListener('message', onMessage);

        file.arrayBuffer().then(buffer => {
            archiveWorker.postMessage({ id, name: file.name, buffer });
        });
    });
}

/**
* Get the number of cases contained in a file, and a label for each of them
* @param {File} file - the selected file (archive or single NIfTI)
* @returns {Promise<{count: number, labels: (string[] | null)}>} - resolve with the number of cases and their labels if found
*/
async function getCaseInfo(file) {
    let nameLower = file.name.toLowerCase();

    // 1. if archive, get the list of files in it
    if (nameLower.endsWith('.zip') || nameLower.endsWith('.7z')) {
        let paths = await listArchiveEntries(file);
        let visitedDicom = new Set();
        let labels = [];

        // Prepare labels for the different cases
        for (let path of paths) {
            const pathLower = path.toLowerCase();

            if (pathLower.endsWith('.nii') || pathLower.endsWith('.nii.gz')) {
                labels.push(pathLower.endsWith('.gz') ? path.slice(0, -3) : path);
            } else if (pathLower.endsWith('.dcm')) {
                let dossier = path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '';

                if (!visitedDicom.has(dossier)) {
                    visitedDicom.add(dossier);
                    labels.push(dossier ? `${dossier}/<DICOM series>` : '<DICOM series>');
                }
            }
        }

        return { count: labels.length || 1, labels: labels.length ? labels : null };
    }

    // 2. if .nii.gz
    if (nameLower.endsWith('.nii.gz')) {
        return { count: 1, labels: [file.name.slice(0, -3)] };
    }

    // 3, else simple .nii / default
    return { count: 1, labels: [file.name] };
}
