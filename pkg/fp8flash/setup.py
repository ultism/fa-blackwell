from setuptools import setup

setup(
    name="fp8flash",
    version="0.1.0",
    description="FA2-compatible fp8 (e4m3) flash-attention prefill for sm90 (prebuilt kernel)",
    packages=["fp8flash"],
    package_data={"fp8flash": ["*.so"]},
    zip_safe=False,
)
