# Awesome-LLMOps Landscape

This directory contains the configuration files and assets for the Awesome-LLMOps landscape website. The landscape is built using [landscape2](https://github.com/cncf/landscape2), a tool developed by CNCF for creating interactive landscapes.

## Overview

The landscape website provides a visual representation of the Awesome-LLMOps ecosystem, categorizing projects into different groups and subcategories. It helps users discover and navigate through the various tools and projects in the LLMOps space.

## Configuration Files

- `data.yml`: Contains the data structure for the landscape, including categories, subcategories, and project items.
- `guide.yml`: Provides descriptive content for categories and subcategories displayed in the landscape guide.
- `settings.yml`: Customizes the appearance and behavior of the landscape website.

## Directory Structure

- `logos/`: Contains logo files for projects and the landscape itself.

## How to Add a New Project

To add a new project to the landscape, follow these steps:

1. **Prepare the project logo**:
   - Create or obtain a logo for the project (PNG or SVG format recommended)
   - Image should be square or have transparent background
   - Place the logo file in the `logos/` directory with a descriptive name

2. **Update `data.yml`**:
   - Find the appropriate category and subcategory for your project
   - Add a new entry under the `items` section with the following format:
   ```yaml
   - name: Project Name
     description: A brief description of the project (1-2 sentences)
     homepage_url: https://github.com/org/repo
     logo: project-logo.png
     repo_url: https://github.com/org/repo
   ```

3. **Update the main README.md**:
   - You can use the `project_request.py` script to automatically add the project to the main README.md
   - Run: `python project_request.py --url https://github.com/org/repo --section "Category" --subsection "Subcategory"`
   - Or manually add the project to the appropriate section in the main README.md

## Running the Landscape Locally

To run the landscape website locally for testing:

1. Use the commands defined in the Makefile:
   ```
   make build     # Build the landscape
   make serve     # Serve the landscape locally
   ```

2. Access the local website at http://127.0.0.1:8000

## Landscape Categories

The landscape currently includes the following main categories:

- **Agents**: Frameworks and tools for building LLM-powered agents
- **Alignment**: Tools for aligning LLMs with human preferences and safety constraints

Additional categories can be added by updating the `data.yml`, `guide.yml`, and `settings.yml` files.