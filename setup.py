import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_benchmark'


def existing(*patterns):
    return [p for pat in patterns for p in glob(pat) if os.path.exists(p)]


setup(
    name=package_name,
    version='0.1.0',
    python_requires='>=3.11',
    packages=find_packages(where='.', include=[f'{package_name}*']),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Shipped benchmark specs, resolved by name at runtime with an in-package fallback
        # so a source tree works without installing.
        (os.path.join('share', package_name, 'specs'), existing('specs/*.yaml')),
        # The behaviour-tree agent type copied into generated scenarios. Without it every
        # generated agent is a builtin `mode: simple` type and Level D cases are inert.
        (os.path.join('share', package_name, 'agent_types'),
         existing('arena_benchmark/scenario/agent_types/*.yaml')),
        (os.path.join('share', package_name, 'promptgen'),
         existing('arena_benchmark/promptgen/*.md', 'arena_benchmark/promptgen/*.yaml')),
        (os.path.join('share', package_name, 'promptgen', 'prompts'),
         existing('arena_benchmark/promptgen/prompts/*.yaml')),
    ],
    package_data={package_name: ['../specs/*.yaml', 'scenario/agent_types/*.yaml', 'promptgen/*.md', 'promptgen/*.yaml', 'promptgen/prompts/*.yaml']},
    install_requires=['setuptools', 'attrs>=22', 'PyYAML>=6'],
    extras_require={
        # Assembly and analysis only. Kept out of install_requires so the spec and
        # orchestration layers stay importable in a bare ROS environment.
        'analysis': ['pyarrow>=14', 'pandas>=2', 'scipy>=1.11'],
        'test': ['pytest>=7', 'numpy>=1.24', 'Pillow>=9', 'scipy>=1.10'],
    },
    zip_safe=True,
    maintainer='Anh Do Duc',
    maintainer_email='anhdd4work@gmail.com',
    description='Benchmark dataset generation for Arena edge-case scenarios.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'arena_bench = arena_benchmark.cli:main',
        ],
    },
)
