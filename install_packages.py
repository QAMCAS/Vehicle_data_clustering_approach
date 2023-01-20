import weka.core.jvm as jvm
from weka.core import packages

jvm.start(packages=True)
packages.install_package("SelfOrganizingMap", version="1.0.3")
items = packages.installed_packages()
for item in items:
    print(item.name + "/" + item.version + "\n  " + item.url)
packages.install_package("multiLayerPerceptrons")
jvm.stop()

