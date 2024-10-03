from setuptools import setup, find_packages

setup(
    name='CyzHub',  # 包名，通常为小写，尽管可以使用大写
    version='0.1.0',  # 包的版本
    author='陈颍州',  # 作者名
    author_email='cyzroot@outlook.com',  # 作者邮箱
    description='深度学习的一些工具（基础阶段的，供参考学习使用）',  # 简短描述
    long_description=open('README.md', 'r', encoding='utf-8').read(),  # 从 README.md 中读取详细描述
    long_description_content_type='text/markdown',  # 描述的格式
    url='https://github.com/CyzSpace/CyzHub',  # 项目的URL
    packages=find_packages(),  # 自动查找项目中的所有包
    classifiers=[
        'Programming Language :: Python :: 3',  # 指定支持的Python版本
        'License :: OSI Approved :: MIT License',  # 许可证类型
        'Operating System :: OS Independent',  # 操作系统兼容性
    ],
    python_requires='>=3.6',  # Python版本要求
)
