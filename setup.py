import setuptools


__version__ = "0.0.0"

REPO_NAME = "ml_credit_card_defaulter"
AUTHOR_USER_NAME = "Radhika5062"
SRC_REPO = "credit_card_defaulter"
AUTHOR_EMAIL = "radhikamaheshwari26@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "Python package for credit card defaulters",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    package_dir = {"":"src"},
    packages = setuptools.find_packages(where="src")
)
