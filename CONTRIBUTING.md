# Contributing

👋 Welcome to InftyAI community !

- [Before you get started](#before-you-get-started)
  - [Code of Conduct](#code-of-conduct)
- [Getting started](#getting-started)
  - [PullRequests](#pull-requests)
  - [Code Review](#code-review)

## Before you get started

### Code of Conduct

Please make sure to read and observe our [Code of Conduct](/CODE_OF_CONDUCT.md) first.

## Getting started

🚀 **All kinds of contributions are welcomed !**

- Fix documents & Typos
- Report & fix bugs
- New features
- Issues & discussions
- ...

### Pull Requests

Pull requests are often called simply "PR".
Please follows the standard [github pull request](https://help.github.com/articles/about-pull-requests/) process.
To submit a proposed change, please develop the code and add new test cases.

### Code Review

To make it easier for your PR to receive reviews, consider the reviewers will need you to:

- Follow [good coding guidelines](https://github.com/golang/go/wiki/CodeReviewComments).
- Write [good commit messages](https://chris.beams.io/posts/git-commit/).
- Break large changes into a logical series of smaller patches which individually make easily understandable changes, and in aggregate solve a broader issue.

### How to Add a New Project

#### Option 1: Using GitHub Issues (Recommended)

The easiest way to add a new project is by creating a Project Request issue:

1. Go to the [Issues tab](https://github.com/InftyAI/Awesome-LLMOps/issues)
2. Click "New Issue" and select "Project Request"
3. Fill out the template with your project information
4. Submit the issue

GitHub Actions workflow will automatically process your request, create a PR, and add the project to the appropriate category.

#### Option 2: Manual Addition

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
   - Add the project to the appropriate category in the main README.md
