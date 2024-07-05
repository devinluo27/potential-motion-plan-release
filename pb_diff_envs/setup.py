from setuptools import find_packages, setup
required = []
extras = {}
setup(
    name="pb_diff_envs",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    extras_require=extras,
    license='MIT',
)