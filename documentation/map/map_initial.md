# Initial map of the repo/website state 

## Architecture

In the production environnement, the website run in a Docker container called `aeyewb` coupled with Traefik (v3.3) reverse proxy in a second docker container `Traefik_aeyeweb` (c.f. `docker-compose.vm.yml`).

### Flow
From internet, access on port :443 to the `Traefik_aeyeweb` container. This container act as a reverse proxy and handle SSL/TLS with Let's Encrypt.

Traefik then act to redirect to port :5000 for the `aeyeweb` container. It is based on Flask + Gunicorn.
`aeyeweb` has ssh access to the HEVS "chacha" server for HPC in the segmentation process. HPC is handled by Slurm.
Also `aeyeweb` mount the `/mnt/filer01` Filer01 volume from the HEVS server for data and logs storage.

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
| GET | `/segmentation` | Yes | Route to Segmentation tool page if user is logged in, else to `/login`|
| POST | `/upload` | No | When `Upload` button pressed : get the files, check and copy them to HPC upload folder|
| POST | `/segment` | No |  When `Segment` button pressed : Launch Slurm job from the uploaded data, wait for completion and get the results |
| GET | `/profile` | Yes | Show user info |
| GET | `/download` | No | When `Download` button pressed : Send the contents of a file to the client |
| GET | `/test-email` | No | send test email to a user |

`/upload`, `/segment`, `/download` aren't requiring authentication as their triggering is only possible if previous authentication was made to access the `/segmentation` dashboard. 

**Remark** : This doesn't prevent direct HTTP requests to access those endpoints. Also `/test-email` is in this case.

## Authentication Flow (using Auth0)

1. The user clicks the "Login" button on the application UI.

2. Authorization Request (`/login` route)

    Generation a nonce and state parameter to prevent CSRF and ID token replay, then storing them both in the user's session.

    Redirection to the Auth0 /authorize endpoint and passing the generated parameters with.

3. User Authentication

    The hosted Auth0 page appear to log with email and password credentials. (Only certains domains are approved for registration (i.e. hevs.ch, chuv.ch,...))

    If no account, the user can sign up, wich then require email verification (`/verify_email` route).

4. Authentication Callback (`/callback` route)

    After successful authentication, the `/callback` route is used. A state and nonce validation is made. The temporary authorization code is exchange for an access token and an ID token. The ID token is parsed to get the user's profile information. The `/segmentation` route is called.

    If email is not verified: `/verify_email` route

5. Logout Flow (`/logout`)
    The local user's session is cleared and redirection to the Auth0 /v2/logout endpoint, which then lead to the  `/` homepage.

## Segmentation

The segmentation process is handled by `getSegmentation` function in `main.py`.

The pipeline is : 
1. Uploading a file with MRI data (supported formats: `.nii.gz` / `.zip` / `.7z` / `.nii`) on the segmentation page.

2. This call the `/upload` route to :
    - Clear previous logs and files before uploading new ones
    - upload the files after some conversion, compression and renaming to the HPC

3. The cleaned data are then used in the `/segment` route as :
        - Update the jobfile template and copy it in HPC folder
        - launch the inference (nnUNet) and wait for result
        - Copy output folder from HPC
        - Copy the logs

4. The results are at disposal for download in the `/download` route

### Inference command 
For reference : 
```bash
ssh {SSH_USER} "sbatch --wait --partition=Dance --account=mattech --qos=normal {jobfile_hpc}"
```
### Slurm template 

The ressources used for the Slurm inference are :
- `nodes`= 1
- `ntasks`= 1
- `cpus-per-task`= 4
- `mem`= 16gb
- `time`= 04:00:00
- `gpu`= 1

*Source :* [nnunet_inference_template.sh](../jobfiles/nnunet_inference_template.sh) 

## Python files

| file | usage |
|---|---|
| `app.py` | initialise the Flask app, register modules (Aut0, Mail) and the routes |
| `config.py` | the one place for hardcoded values |
| `routes.py` | HTTP routes endpoints |
| `main.py` | `getSegmentation()` — segmentation process using slurm on chacha server|
| `utils.py` | helpers functions (i.e. DICOM to NIfTI, Auth0, log, docker,...) |
| `run.py` | entry point to create Flask instance |