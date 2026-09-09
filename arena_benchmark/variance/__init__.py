"""Measures of how different the dataset's worlds and test cases are from one another.

`world_features` turns a world's map (walls plus furniture) and zones into a small feature
vector and measures the spread of a set of worlds in that space; `diversity` does the same for
prompt sets and the scenarios realised from them.
"""
