from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithoutLocalValidation(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            module
            for module in modules
            if not (module[0] == "examples" and module[1] == "local_validation")
        ]


setup(cmdclass={"build_py": BuildPyWithoutLocalValidation})
