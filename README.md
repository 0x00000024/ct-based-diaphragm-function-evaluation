# Diaphragmatic Function Evaluation Experimental Manual

This manual is for setting up and running the experimental program for diaphragmatic function evaluation, which uses several tools including OpenCV, MATLAB, and CGAL. The main program is cross-platform and can be run on Windows, macOS, and Linux.

## Contents

- [OpenCV Environment Setup](#opencv-environment-setup)
- [MATLAB Environment Setup](#matlab-environment-setup)
- [CGAL Environment Setup](#cgal-environment-setup)
- [Running the Experimental Program](#running-the-experimental-program)

## OpenCV Environment Setup

The program contains manually run programs. It is recommended to use automation software similar to Keyboard Maestro to speed up the experiment progress.

1. Switch the branch to main: `git checkout main`
2. Download Miniconda and modify the executable path of conda in the `deploy_env.sh` script. For example:

    ```bash
    conda="/opt/homebrew/Caskroom/miniconda/base/bin/conda"
    ```

3. Use the `deploy_env.sh` script to deploy Python and OpenCV development environments.

    ```bash
    bash deploy_env.sh
    ```

## MATLAB Environment Setup

1. Download MATLAB. Install all packages during the installation process.
2. MATLAB command line program path reference: `/Applications/MATLAB_R2mataa021a.app/bin/maci64/MATLAB`

## CGAL Environment Setup

1. Switch the branch to CGAL-docker: `git checkout cgal-docker`
2. Update the `ROOT_PASSWORD` and `PUBLIC_KEY` environment variables in the `docker-compose.yml` file.
3. Deploy the CGAL environment with the following command:

    ```bash
    docker compose -f docker-compose.yml up -d
    ```

The container's sshd port 22 is mapped to the host port 2200. The container's xrdp port 3390 is mapped to the host port 33890.

4. Connect to the container with a tool that supports the xrdp protocol, such as Jump Desktop.

   ![Jump Desktop 1](https://i.imgur.com/NWTCyZZ.png)
   ![Jump Desktop 2](https://i.imgur.com/SvFQQX8.png)

5. Download the project file into the container and open it with CLion.

   ![CLion 1](https://i.imgur.com/xKNWMHM.png)

    ```bash
    cd /root
    git clone https://github.com/0x00000024/ct-based-diaphragm-function-evaluation
    cd ct-based-diaphragm-function-evaluation
    git checkout auto
    bash /opt/clion-2022.1.2/bin/clion.sh .
    ```

6. Set the CGAL and C++ Boost variables in the CMake options. `-DCGAL_DIR="/usr/local/lib/cgal" -DBOOST_ROOT="/usr/local/lib/boost"`

   ![CLion 2](https://i.imgur.com/c9hLGUF.png)

7. Compile two executable programs: `5-8-auto` (Used to generate mask and surface 3D point cloud.) and `11-area` (Used to calculate the surface area.) The path to the compiled executable is as follows:

   For `5-8-auto`:

    ```bash
    mkdir --parents /root/ct
    git checkout auto
    /opt/clion-2022.1.2/bin/cmake/linux/bin/cmake --build /root/ct-based-diaphragm-function-evaluation/cmake-build-debug --target main -j 44
    mv /root/ct-based-diaphragm-function-evaluation/cmake-build-debug/main /root/ct/5-8-auto
    ```

   For `11-area`:

    ```bash
    git checkout area
    /opt/clion-2022.1.2/bin/cmake/linux/bin/cmake --build /root/ct-based-diaphragm-function-evaluation/cmake-build-debug --target main -j 44
    mv /root/ct-based-diaphragm-function-evaluation/cmake-build-debug/main /root/ct/11-area
    ```

## Running the Experimental Program

1. Add the patient IDs you want to test to the `test/input.csv` file in the project directory.

   ![CSV file](https://i.imgur.com/zPVhI0s.png)

2. Put the image to be tested into the `images` folder.

   ![Image folder](https://i.imgur.com/6vzoEI9.png)

3. Update the MATLAB executable path in the `src/respiration/surface_fitting.py` file.

   ![MATLAB path](https://i.imgur.com/2DGmKoL.png)

4. Change CGAL server info in the `src/respiration/base.py` file.

   ![CGAL server](https://i.imgur.com/wvfemrK.png)

5. The final results generated for each patient are located in `results/patient_id/in_or_ex`.

   ![Result file](https://i.imgur.com/m8jZkAM.png)

6. To run the program, first activate the Conda environment, then execute the main Python script:

    ```bash
    conda activate ct-3.10
    python3 main.py
    ```

## Conclusion

After following these steps, you should have the necessary environment setup and be able to run the diaphragmatic function evaluation program. Happy researching!

## Relevant Links

- [main](https://github.com/0x00000024/ct-based-diaphragm-function-evaluation/tree/main)
- [cgal-docker](https://github.com/0x00000024/ct-based-diaphragm-function-evaluation/tree/cgal-docker)
- [auto](https://github.com/0x00000024/ct-based-diaphragm-function-evaluation/tree/auto)
- [area](https://github.com/0x00000024/ct-based-diaphragm-function-evaluation/tree/area)
