# Breast Cancer Ultrasound Segmentation

An AI-powered diagnostic tool for analyzing breast ultrasound images. This application leverages deep learning models to segment regions of interest in ultrasound images and generate detailed clinical reports. The tool is built with Streamlit, TensorFlow, OpenCV, and several other libraries to provide an interactive and user-friendly experience.

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Features

- **User Authentication:** Secure login interface for accessing the application.
- **Image Processing:** Supports both DICOM and standard image formats (PNG, JPG, JPEG).
- **Deep Learning Integration:** Uses a TensorFlow model for segmentation of ultrasound images.
- **Annotation Tools:** Provides drawing and annotation capabilities via `streamlit-drawable-canvas`.
- **Clinical Metrics:** Automatically calculates important clinical metrics including tumor area, maximum diameter, irregularity index, and BI-RADS score.
- **Longitudinal Analysis:** Allows users to compare current studies with prior examinations.
- **Report Generation:** Automatically generates comprehensive reports in both Markdown and PDF formats.
- **Interactive Visualization:** Displays original images, segmentation results, heatmaps, and clinical reports using Streamlit tabs.
- **Customizable Interface:** Theme selection and various settings enable a tailored user experience.
