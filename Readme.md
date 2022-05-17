# point-cloud-visualizer

To run the program, open this folder in VSCode, and run the following commands:

> curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
>
> nvm install --lts
>
> npm install
>
> export license_key="your_license_key_here"
>
> sed -i.bak "s/SciChartSurface.setRuntimeLicenseKey();/SciChartSurface.setRuntimeLicenseKey('${license_key}');/g" src/index.js
>
> npm start 

Then visit https://localhost:8080 in your web browser!
