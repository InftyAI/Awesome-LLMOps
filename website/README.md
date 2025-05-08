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

- **Inference**: Tools and platforms for deploying and serving LLMs
- **Orchestration**: Tools for orchestrating LLM workflows and agents
- **Runtime**: Runtime environments and tools for LLM applications
- **Training**: Tools and frameworks for training and fine-tuning LLMs
- **MCP**: Model Context Protocol related tools

Additional categories can be added by updating the `data.yml`, `guide.yml`, and `settings.yml` files.
