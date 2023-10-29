"""Setup Script"""
from typing import Tuple, List, Dict
import sys
import re
from subprocess import call
import pkg_resources

# Constants - Only change if necesarry
MIN_PY_VERSION = 3
MIN_PY_SUBVERSION = 9
REQUIREMENTS_PATH = 'requirements.txt'
PATTERN = r"^([^>=~<# ]+)\s*((>=|==|~=|<=|<|>)?\s*\d+(\.\d+)*([abrc]\d+"\
          r"|\.post\d+|\.dev\d+)?(\+\w+(\.\w+)*)?)?\s*(#\s+.*)?$"

def check_python_version(
    min_version:int,
    min_subversion:int
) -> bool:
    """Check if the current Python version meets the minimum required version.

    Args:
        min_version (int): The minimum required major version of Python.
        min_subversion (int): The minimum required minor version of Python.

    Returns:
        bool: True if the current Python version meets the minimum requirements, False otherwise.
    """
    major, minor = sys.version_info[:2]
    if major < min_version or\
       (major == min_version and minor < min_subversion):
        print(f" - 🚨  Python {min_version}.{min_subversion} or higher is required.")
        return False
    return True

def split_version_string(
    s:str
) -> Tuple[str, str, str]:
    """Split the given version string into package name, version specifier, and optional comment.

    Args:
        s (str): The version string to be split.

    Returns:
        Tuple[str, str, str]: A tuple containing the package name, version specifier, and optional
                              arbitrary command line argument(s) placed in the comments after the
                              package name and version specification for the line in the
                              `requirements.txt` file. Example: `captum~=0.6.0 # --no-deps`.

    Note:
        - If the version string does not match the specified pattern, None is returned.
        - The package name, version specifier, and optional arguments are stripped of leading and
          trailing whitespace.
        - The version specifier and optional arguments are empty strings if not present in the
          version string.
        - The optional comment is stripped of the leading '# ' characters.
        - If the optional comment starts with '--' or '-', it is considered as an invalid
          option and set to None.
    """
    match = re.match(PATTERN, s)
    if match:
        package_name = match.groups()[0].strip()
        version_specifier = match.groups()[1].strip() if match.groups()[1] is not None else ""
        options = match.groups()[7][2:].strip() if match.groups()[7] is not None else ""
        if (len(options) < 2) or (options[:2]!="--") or (options[:1]!="-"):
            options = None
        return package_name, version_specifier, options
    else:
        return None

def parse_requirements_file(
    requirements_path:str
) -> Tuple[List[Tuple], List[Tuple]]:
    """Parse the requirements file and separate the packages into two lists: packages and
       git_packages.

    Args:
        requirements_path (str): The path to the requirements file.

    Returns:
        Tuple[List, List]: A tuple containing two lists. The first list contains the parsed
                           packages (excluding git packages), and the second list contains
                           the parsed git packages.

    Example:
        parse_requirements_file('requirements.txt')

    Note:
        - The requirements file should be a text file containing package names and version
          specifiers, with each package on a separate line.
        - Packages that start with 'git+' are considered git packages and will be included
          in the git_packages list.
        - The function uses the split_version_string() function to parse each requirement
          into a package name and version specifier.
    """

    # Load the requirements file
    with open(requirements_path, 'r', encoding="utf-8") as file:
        requirements = [line.strip() for line in file if line.strip()]

    # Parse the requirements into package names and version specifiers
    packages = [split_version_string(req) for req in requirements if "git+" not in req]
    packages = [item for item in packages if item is not None]
    git_packages = [split_version_string(req) for req in requirements if "git+" in req]
    git_packages = [item for item in git_packages if item is not None]

    return packages, git_packages

def extract_package_name(
        pkg_name: str
    ) -> str:
    """Removes content inside brackets and converts the package name to lowercase.

    Args:
        pkg_name (str): The package name.

    Returns:
        str: The cleaned package name in lowercase.
    """
    # Remove content inside brackets
    clean_name = re.sub(r'\[.*?\]', '', pkg_name)
    # Convert to lowercase
    return clean_name.lower()

def get_installed_packages() -> Dict:
    """Returns a dictionary of installed packages.

    Returns:
        dict: A dictionary where the keys are the names of the installed packages and
              the values are their versions.
    """
    installed_packages = {extract_package_name(pkg.key): pkg.version\
                          for pkg in list(pkg_resources.working_set)}

    return installed_packages

def get_missing_packages(
    installed_packages:Dict,
    packages:List[Tuple],
    git_packages:List[Tuple]
) -> List[Tuple]:
    """Get missing packages to install.

    Args:
        installed_packages (Dict): A dictionary of installed packages.
        packages (List[Tuple]): A list of packages to check for installation.
        git_packages (List[Tuple]): A list of packages to always install from GIT.

    Returns:
        List[Tuple]: A list of packages to install.
    """
    packages_to_install = []

    for pkg, version_spec, options in packages:
        if extract_package_name(pkg) not in installed_packages:
            print(f" - ⚠️  {pkg} is not installed")
            packages_to_install.append((f"{pkg}{version_spec}", options))
        else:
            # Check version requirements, pkg_resources will raise a VersionConflict if the version
            # doesn't match
            try:
                pkg_resources.require(f"{pkg}{version_spec}")
            except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound) as e:
                print(f" - ⚠️  {pkg}{version_spec} is getting installed because: {e}")
                packages_to_install.append((f"{pkg}{version_spec}", options))

    for pkg, version_spec, options in git_packages:
        pkg_name = pkg.replace('git+https://github.com/','').replace('.git', '')
        print(f" - ⚠️  always install from GIT: {pkg_name}")
        packages_to_install.append((f"{pkg}", options))

    return packages_to_install

def install_packages(
    packages_to_install:List[Tuple]
) -> None:
    """Installs the specified packages using pip.

    Args:
        packages_to_install (List[Tuple]): A list of tuples containing the package name/URL
                                           and options (if any).

    Returns:
        None

    Example:
        install_packages([
            ('git+https://github.com/username/reponame/package1.git', None),
            ('captum~=0.6.0', '--upgrade'),
        ])
    """
    if packages_to_install:
        for package, options in packages_to_install:
            pkg_name = package.replace('git+https://github.com/','').replace('.git', '')
            print(f"\r\n - ⚙️  installing {pkg_name}...")
            if options is None:
                call(["pip", "install", package])
            else:
                opts = options.split()
                call(["pip", "install"] + opts + [package])
    else:
        print(" - 🙌 All required packages are already installed!")

def main() -> None:
    """Main function to check and install required packages.

    This function checks the Python version and exits if it does not meet the minimum required
    version.
    It then parses the requirements file to get the required packages and git packages.
    Next, it retrieves the installed packages.
    After that, it identifies the missing packages by comparing the installed packages with
    the required packages.
    Finally, it installs the missing packages.

    Returns:
        None
    """
    if not check_python_version(MIN_PY_VERSION, MIN_PY_SUBVERSION):
        sys.exit(1)

    req_packages, req_git_packages = parse_requirements_file(REQUIREMENTS_PATH)

    inst_packages = get_installed_packages()

    missing_packages = get_missing_packages(inst_packages,
                                               req_packages, req_git_packages)

    install_packages(missing_packages)

if __name__ == '__main__':
    main()
