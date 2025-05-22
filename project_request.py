import argparse
import os
import re
import requests
import sys
import yaml
from typing import Tuple, Dict, List, Any, Optional
from urllib.parse import urlparse

# Constants
README_PATH = 'README.md'
DATA_YML_PATH = 'website/data.yml'
LOGOS_DIR = 'website/logos'
# Categories that should only be added to README.md, not to website/data.yml
README_ONLY_CATEGORIES = ["MCP/MCP Server", "MCP/MCP Client"]

def parse_github_url(url: str) -> Tuple[str, str]:
    """Extract owner and repository name from a GitHub URL.
    
    Args:
        url: A GitHub repository URL (e.g., https://github.com/owner/repo)
        
    Returns:
        A tuple containing (owner, repo) strings
        
    Raises:
        ValueError: If the URL is not a valid GitHub repository URL
    """
    parsed_url = urlparse(url)
    if 'github.com' not in parsed_url.netloc:
        raise ValueError(f"Not a GitHub URL: {url}")
    
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {url}")
    
    owner, repo = path_parts[0], path_parts[1]
    return owner, repo


def get_repo_info(owner: str, repo: str) -> Dict[str, Any]:
    """Fetch repository information from GitHub API.
    
    Args:
        owner: GitHub repository owner/organization name
        repo: GitHub repository name
        
    Returns:
        Dictionary containing repository information from GitHub API
        
    Raises:
        Exception: If the API request fails
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch repository info: {response.status_code} {response.text}")
        
    return response.json()


def generate_entry(repo_url: str, project_name: str) -> Tuple[str, str]:
    """Generate formatted entry for README.md.
    
    Args:
        repo_url: GitHub repository URL
        project_name: The project name
        
    Returns:
        A tuple containing:
            - project_name: The name of the project
            - entry: A formatted markdown string for the README entry
    """
    # Parse GitHub URL
    owner, repo = parse_github_url(repo_url)
    
    # Get repository description
    repo_info = get_repo_info(owner, repo)
    description = repo_info.get('description', '')
    
    # Generate shields.io URLs
    stars_badge = f"![Stars](https://img.shields.io/github/stars/{owner}/{repo}.svg?style=flat&color=green)"
    contributors_badge = f"![Contributors](https://img.shields.io/github/contributors/{owner}/{repo}?color=green)"
    last_commit_badge = f"![LastCommit](https://img.shields.io/github/last-commit/{owner}/{repo}?color=green)"
    
    # Format the entry
    entry = f"* **[{project_name}]({repo_url})**: {description} {stars_badge} {contributors_badge} {last_commit_badge}"
    
    return project_name, entry


def find_category(content: str, category: str) -> Tuple[int, int, List[str]]:
    """Find the specified category in the README content.
    
    Args:
        content: The full content of the README.md file
        category: The name of the category to find (e.g., "framework" or "orchestration/workflow")
                 Can include a path with '/' as separator for nested categories
        
    Returns:
        A tuple containing:
            - category_start_line: The line number where the category starts
            - category_end_line: The line number where the category ends
            - lines: List of all lines in the content
            
    Raises:
        ValueError: If the specified category is not found
    """
    
    # Define patterns for different category levels
    category_patterns = [
        # Main category (##)
        re.compile(r'##\s+([^\n]+)'),
        # Subcategory (###)
        re.compile(r'###\s+([^\n]+)')
    ]
    
    lines = content.split('\n')
    category_path = [c.strip().lower() for c in category.split('/')]
    
    # If we have a path with multiple levels, we need to find each level
    if len(category_path) > 1:
        current_path = []
        current_level = 0
        category_start_line = -1
        category_end_line = -1
        
        for i, line in enumerate(lines):
            # Check if this line starts a category
            for pattern in category_patterns:
                match = pattern.match(line)
                if match:
                    # Get the heading level (## = 2, ### = 3)
                    heading_level = line.count('#')
                    category_name = match.group(1).strip().lower()
                    
                    # If we're at a level we're tracking
                    if heading_level - 2 <= len(current_path):
                        # If we're at a lower level than current, pop levels
                        while heading_level - 2 < len(current_path):
                            current_path.pop()
                        
                        # If we're at a new level, add it
                        if heading_level - 2 == len(current_path):
                            current_path.append(category_name)
                        # If we're at the same level, replace the last item
                        else:
                            current_path[-1] = category_name
                        
                        # Check if the current path matches our target path
                        if len(current_path) == len(category_path) and all(a == b for a, b in zip(current_path, category_path)):
                            category_start_line = i
                        # If we already found our category and encounter another at the same or higher level, that's the end
                        elif category_start_line != -1 and category_end_line == -1 and heading_level - 2 <= len(category_path) - 1:
                            category_end_line = i
                            break
    else:
        # Original single-level category search
        category_start_line = -1
        category_end_line = -1
        current_category = ""
        
        for i, line in enumerate(lines):
            # Check if this line starts a category
            for pattern in category_patterns:
                match = pattern.match(line)
                if match:
                    # If we already found our category, this new category marks the end
                    if category_start_line != -1 and category_end_line == -1:
                        category_end_line = i
                        break
                    
                    # Check if this is the category we're looking for
                    current_category = match.group(1).strip().lower()
                    if current_category == category_path[0]:
                        category_start_line = i
                        break
    
    # If we found the start but not the end, the category goes to the end of the file
    if category_start_line != -1 and category_end_line == -1:
        category_end_line = len(lines)
    
    if category_start_line == -1:
        raise ValueError(f"Category '{category}' not found in README.md")
    
    return category_start_line, category_end_line, lines


def insert_entry(lines: List[str], category_start_line: int, category_end_line: int, project_name: str, new_entry: str) -> List[str]:
    """Insert the new entry in alphabetical order within the category.
    
    Args:
        lines: List of all lines in the README.md file
        category_start_line: The line number where the category starts
        category_end_line: The line number where the category ends
        project_name: The name of the project
        new_entry: The formatted entry to insert
        
    Returns:
        Updated list of lines with the new entry inserted in alphabetical order
    """
    
    # Find the correct position to insert the new entry
    insert_position = category_end_line
    last_entry_position = -1
    
    # Skip the category header
    for i in range(category_start_line + 1, category_end_line):
        line = lines[i]
        # Check if this line is an entry
        entry_name_match = re.search(r'\*\s+\*\*\[([^\]]+)\]', line)
        if entry_name_match:
            last_entry_position = i
            entry_name = entry_name_match.group(1).lower()
            # If the new entry comes before this entry alphabetically (case-insensitive comparison)
            if project_name.lower() < entry_name:
                insert_position = i
                break
    
    # If we're inserting at the end of the category
    if insert_position == category_end_line:
        # Always insert after the last entry
        insert_position = last_entry_position + 1
    
    # Insert the new entry at the determined position
    lines.insert(insert_position, new_entry)
    
    return lines


def update_website(category: str, project_name: str, repo_url: str, homepage_url: str, logo_url: str = None, logo_name: str = None) -> bool:
    """Update the website data.yml file and download the logo.
    
    Args:
        category: The name of the category to add the entry to (e.g., "Inference Engine")
        project_name: The name of the project
        repo_url: The GitHub repository URL
        logo_url: URL to the project logo (optional, default.png will be used if not provided)
        homepage_url: Custom homepage URL
        logo_name: Optional custom logo filename
        
    Returns:
        True if the update was successful, False otherwise
    """
    try:
        # Load the data.yml file
        with open(DATA_YML_PATH, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        # Parse GitHub URL
        owner, repo = parse_github_url(repo_url)
        
        # Get repository information
        repo_info = get_repo_info(owner, repo)
        description = repo_info.get('description', '')

        # Process logo
        logo_filename = None

        # If no logo_url is provided, use default.png
        if not logo_url:
            logo_filename = "default.png"
        else:
            # If logo_name is provided, use it directly
            if logo_name:
                logo_filename = logo_name
            else:
                # Extract filename from URL
                parsed_url = urlparse(logo_url)
                original_filename = os.path.basename(parsed_url.path)
                file_ext = os.path.splitext(original_filename)[1].lower()
                
                # Create a sanitized filename based on project name
                sanitized_name = project_name.lower().replace(' ', '-')
                sanitized_name = re.sub(r'[^\w\-]', '', sanitized_name)
                logo_filename = f"{sanitized_name}{file_ext}"
            logo_path = os.path.join(LOGOS_DIR, logo_filename)
            
            # Download the logo
            response = requests.get(logo_url, stream=True)
            response.raise_for_status()
            
            with open(logo_path, 'wb') as logo_file:
                for chunk in response.iter_content(chunk_size=8192):
                    logo_file.write(chunk)
            
            print(f"Successfully downloaded logo to {logo_path}")

        # Parse the category path
        category_path = [c.strip().lower() for c in category.split('/')]
        target_category = category_path[-1]  # Use the last part as the actual category name
        
        # Find the appropriate category and subcategory
        for maincategory in data['categories']:
            for subcategory in maincategory['subcategories']:
                if subcategory['name'].lower() == target_category:
                    # Create new item entry
                    new_item = {
                        'name': project_name,
                        'description': description,
                        'homepage_url': homepage_url,
                        'logo': logo_filename,
                        'repo_url': repo_url
                    }
                    
                    # Add the new item to the subcategory
                    subcategory['items'].append(new_item)
                    
                    # Sort items by name
                    subcategory['items'] = sorted(subcategory['items'], key=lambda x: x['name'].lower())
                    
                    # Write the updated data back to the file
                    with open(DATA_YML_PATH, 'w', encoding='utf-8') as file:
                        yaml.dump(data, file, sort_keys=False, default_flow_style=False, allow_unicode=True)
                    
                    return True

    except Exception as e:
        print(f"Error updating website data: {str(e)}")
        return False


def update_readme(category: str, project_name: str, new_entry: str) -> bool:
    """Update the README.md file with the new entry.
    
    Args:
        category: The name of the category to add the entry to
        project_name: The name of the project
        new_entry: The formatted entry to add
        
    Returns:
        True if the update was successful, False otherwise
    """

    try:
        with open(README_PATH, 'r', encoding='utf-8') as file:
            content = file.read()
        
        category_start_line, category_end_line, lines = find_category(content, category)
        updated_lines = insert_entry(lines, category_start_line, category_end_line, project_name, new_entry)
        
        # Write the updated content back to the file
        with open(README_PATH, 'w', encoding='utf-8') as file:
            file.write('\n'.join(updated_lines))
        
        print(f"Successfully added {project_name} to {category} category in README.md")
        return True
    
    except Exception as e:
        print(f"Error updating README.md: {str(e)}")
        return False


def is_readme_only(category: str) -> bool:
    """Check if a project should only be added to README.md and not to website/data.yml.
    
    Args:
        category: The category of the project
        
    Returns:
        True if the project should only be added to README.md, False otherwise
    """
    # Normalize the category for case-insensitive comparison
    normalized_category = category.strip().lower()
    
    # Check if the category is in the README_ONLY_CATEGORIES list
    for readme_only_category in README_ONLY_CATEGORIES:
        if normalized_category == readme_only_category.lower():
            return True
    
    return False


def main() -> None:
    """Main function to parse arguments and execute the script.
    
    Command line arguments:
        --category/-c: The category to add the project to (e.g., "Inference/Inference Engine", "Orchestration/Workflow")
                      Can include a path with '/' as separator for nested categories
        --repo_url/-r: The GitHub repository URL
        --name/-n: Custom project name
        --logo_url/-l: URL to the project logo (optional for README_ONLY_CATEGORIES)
        --homepage_url/-hu: Project homepage URL (optional for README_ONLY_CATEGORIES)
        --logo_name/-ln: Optional custom logo filename

    Example:
        python project_request.py \
            --category "Inference/Inference Engine" \
            --repo_url https://github.com/google/adk-python \
            --name "Agent Development Kit (ADK)" \
            --logo_url https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png \
            --homepage_url https://google.github.io/adk-docs
    """
    parser = argparse.ArgumentParser(description='Add a new project to the README.md file and update the website data.')
    parser.add_argument('--category', '-c', required=True, help='The category to add the project to (e.g., "Inference Engine", "Agent", "Orchestration/Workflow"). Can include a path with "/" as separator for nested categories.')
    parser.add_argument('--repo_url', '-r', required=True, help='The GitHub repository URL')
    parser.add_argument('--name', '-n', required=True, help='Custom project name')
    parser.add_argument('--logo_url', '-l', required=False, help='URL to the project logo (optional for MCP-related projects)')
    parser.add_argument('--homepage_url', '-hu', required=False, help='Custom homepage URL (optional for MCP-related projects)')
    parser.add_argument('--logo_name', '-ln', required=False, help='Optional custom logo filename')
    
    args = parser.parse_args()
    
    try:
        # Generate the entry for README.md
        project_name, entry = generate_entry(args.repo_url, args.name)
        
        # Update the README.md file
        readme_success = update_readme(args.category.lower(), project_name, entry)
        
        if not readme_success:
            print("Failed to update README.md")
            sys.exit(1)
        
        # Check if the project is in README_ONLY_CATEGORIES
        readme_only = is_readme_only(args.category)
        
        if readme_only:
            print(f"Category '{args.category}' is in README_ONLY_CATEGORIES. Skipping website data update.")
            website_success = True
        else:
            # For non-README_ONLY_CATEGORIES projects, warn if logo_url is not provided
            if not args.logo_url:
                print("Warning: No logo URL provided, using default.png")
            # For non-README_ONLY_CATEGORIES projects, homepage_url is required
            if not args.homepage_url:
                print("Error: --homepage_url is required for projects not in README_ONLY_CATEGORIES")
                sys.exit(1)
                
            # Update website
            website_success = update_website(args.category, args.name, args.repo_url, args.homepage_url, args.logo_url,args.logo_name)
            
            if not website_success:
                print("Failed to update website data")
                sys.exit(1)
            else:
                print(f"Successfully updated website data for {project_name}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    print(f"Successfully added {project_name} to the {args.category} category")

if __name__ == '__main__':
    main()
