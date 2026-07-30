# Final map of the repo/website at handout state

This is an updated version of [map_initial.md](map_initial.md) with the details of changes made since the start of my bachelor thesis.

## Architecture

Slight change from [the initial map](map_initial.md#architecture) : in production, the website runs in a Docker container called `aeyewb` coupled with a Traefik (updated version to v3.7.4 ) reverse proxy in a second container `Traefik_aeyeweb`.

### Flow
From internet, access on port :443 to the `Traefik_aeyeweb` container. This container acts as a reverse proxy and handles SSL/TLS with Let's Encrypt.

Traefik then redirects to port :5000 for the `aeyeweb` container. It is based on Flask + Gunicorn.
`aeyeweb` has ssh access to the HEVS "chacha" server for HPC in the segmentation process. HPC is handled by Slurm.
Also `aeyeweb` mount the `/mnt/filer01` Filer01 volume from the HEVS server for data and logs storage.

**Change since the initial map** : the pipeline is now multi-user safe (tested with 4 concurent users). Every local and HPC path used during upload,inference and download phases is scoped per user , instead of a single shared set of folders. (c.f. [Multi-users segmentation](../contributions/1.2_multi_users_segmentation.md))

## URL routes

| HTTP method | Path | authentication required | Description |
|---|---|---|---|
| GET | `/` | No | Welcome page |
| GET | `/callback` | No | Auth0 callback |
| GET | `/verify_email` | No | Should only appear after a sign up (email known but not verified) |
| GET | `/login` | No | When `Login` button pressed : generate nonce and state for session then redirects to Auth0 authorize URL |
| GET | `/logout` | No | When `Logout` button pressed : clears session then redirects to Auth0 logout |
| GET | `/users` | No | Global stats page from all the website usage data |
| GET | `/about` | No | About page (fund, team and partners) |
| GET | `/faq` | No | FAQ page |
| GET | `/segmentation` | Yes | Route to Segmentation tool page if user is logged in, else to `/login`. Also cancels any Slurm job left running for that user |
| POST | `/upload` | Yes | When `Upload` button pressed : clears the user's previous data, saves the files, extracts their metadata and copies them to the HPC input directory |
| POST | `/segment` | Yes | When `Segment` button pressed : crops each case into left/right quadrants, launches the Slurm job, uncrops and merges the results, zips the output and starts a background copy to Filer01 |
| GET | `/profile` | Yes | Show user info |
| GET | `/download` | Yes | Send the user's result ZIP to the client |
| GET | `/test-email` | Yes | Send test email to a user |
| GET | `/result/<filename>` | Yes | Serve a `.nii.gz` result file from the user's download folder |
| GET | `/result-image/<filename>` | Yes | Serve a generated image (e.g. axial length plot) from the user's download folder |
| GET | `/display-image/<filename>` | Yes | Serve a generated image (e.g. violin plot) from the user's visualisation folder |
| POST | `/biomarkers` | Yes | Run volumetry + axial length extraction, reference comparison and outlier detection for the given cases, then add the .CSV in ZIP |
| GET | `/license` | No | Serve `LICENSE.txt` |
| POST | `/reset` | Yes | Cancel any running Slurm job and delete the user's uploaded files and segmentation results |

## Authentication Flow (using Auth0)

Same as [the initial map](map_initial.md#authentication-flow-using-auth0), with one addition: the API routes (`/upload`, `/segment`, `/biomarkers`, `/reset`) now also require authentication (change made by Jaime Barranco).

## Segmentation

The segmentation process is handled by `getSegmentation` in `main.py` and called from the `/segment` route in `routes.py`. Since the initial map, the pipeline is user-scoped and includes quadrant cropping.

The pipeline is now :
1. Upload a file with MRI data (supported formats: `.nii.gz` / `.zip` / `.7z` / `.nii`) on the segmentation page.

2. This call the `/upload` route to :
    - Clear the user's previous logs and files
    - Save the files locally and extract their NIfTI metadata (c.f. [Metadata](../contributions/2.1_metadata.md))
    - upload the files after some conversion, compression and renaming to the HPC

3. `/segment` route :
    - Crop each case into left/right eye quadrants (c.f. [Quadrant segmentation](../contributions/3.1_quadrant_segmentation.md)) and send the cropped versions to the HPC
    - update the jobfile template and copy it to the HPC
    - launch the inference (nnUNet) and wait for the result
    - copy the output folder back from the HPC
    - uncrop and merge the left/right results into a combined volume (`_left`, `_right`, `_both`)
    - copy the logs, zip the results folder

4. Results are at disposal for download via `/download`, or individually via `/result` and `/result-image`.

5. On the results page, the user can additionally trigger `/biomarkers` to extract volumetry and axial length per eye, compare them to the reference dataset and detect outliers (c.f. [Biomarkers extraction](../contributions/3.2_biomarkers_extraction.md) and [Statistical analysis](../contributions/4.1_statistical_analysis.md)).

### Inference command
Unchanged, c.f. [the initial map](map_initial.md#inference-command).

### Slurm template
Unchanged, c.f. [the initial map](map_initial.md#slurm-template).

## Added features since the initial map

A brief summary of the added features.

### Multi-users segmentation (MUS)
All local and HPC paths are now generated per user. This removes the race conditions / data-mixing issues that existed when several users ran the pipeline concurrently. (c.f. [1.2_multi_users_segmentation.md](../contributions/1.2_multi_users_segmentation.md))

### Metadata extraction
NIfTI header metadata is extracted at upload time and shown for each case after upload and in the segmentation result display. (c.f. [2.1_metadata.md](../contributions/2.1_metadata.md))

### Segmentation results display
Segmentation outputs are displayed directly in the browser instead of only being available as a downloadable ZIP. (c.f. [2.2_segmentation_results.md](../contributions/2.2_segmentation_results.md))

### Quadrant segmentation
Cases are cropped into left/right eye quadrants before inference (smaller volume so faster inference), then uncropped and merged back into a combined result. (c.f. [3.1_quadrant_segmentation.md](../contributions/3.1_quadrant_segmentation.md))

### Biomarkers extraction
Volumetry and axial length are computed per eye, displayed after extraction and results are included in the downloadable ZIP. (c.f. [3.2_biomarkers_extraction.md](../contributions/3.2_biomarkers_extraction.md))

### Statistical analysis
Case's biomarkers are compared against a reference dataset. A dataset formed by all case's extracted biomarkers is also compared to the reference dataset. (c.f. [4.1_statistical_analysis.md](../contributions/4.1_statistical_analysis.md))

## Python files

| file | usage |
|---|---|
| `app.py` | initialise the Flask app, register modules (Auth0, Mail) and the routes |
| `config.py` | the one place for hardcoded values |
| `routes.py` | HTTP routes endpoints |
| `main.py` | `getSegmentation()` — segmentation process using Slurm on the chacha server, now user-scoped via `UserPaths` |
| `utils.py` | helper functions (DICOM→NIfTI, Auth0, logs, docker, HPC transfer, per-user paths, ...) |
| `user_paths.py` | `UserPaths` dataclass — every local and HPC path a user's segmentation run needs |
| `run.py` | entry point to create the Flask instance |
| `package/quadrant_segmentation/quadrant.py` | crop/uncrop/merge eye quadrants |
| `package/biomarkers/biomarkers.py` | volumetry and axial length computation |
| `package/biomarkers/visualisations.py` | axial length ray plot |
| `package/biomarkers/al_data.py` | `ALData` container for axial length ray sampling |
| `package/statistical_analysis/analysis.py` | reference dataset loading, means/std/IQR bounds |
| `package/statistical_analysis/visualisations.py` | volumetry and axial length violin plots |

