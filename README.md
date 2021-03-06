# ReQueST — (Re)ward (Que)ry (S)ynthesis via (T)rajectory Optimization

[ReQueST](https://arxiv.org/abs/1912.05652) is a [reward modeling](https://arxiv.org/abs/1811.07871)
algorithm that asks the user for feedback on hypothetical trajectories synthesized using a
pretrained model of the environment dynamics, instead of real trajectories generated by rolling out
a partially-trained agent in the environment. Compared to
[previous](https://deepmind.com/blog/article/learning-through-human-feedback)
[approaches](https://arxiv.org/abs/1811.06521), this enables

1.  training more robust reward models that work off-policy,
2.  learning about unsafe states without visiting them, and
3.  better query-efficiency through the use of active learning.

This codebase implements ReQueST in three domains:

1.  An MNIST classification task.
2.  A simple state-based 2D navigation task.
3.  The
    [Car Racing task from the OpenAI Gym](https://gym.openai.com/envs/CarRacing-v0/).

All experiments use labels from a synthetic oracle instead of a real human.

## Usage

1.  Setup the Anaconda virtual environment with `conda env create -f
    environment.yml`
3.  Patch the gym car_racing environment by running `bash
    apply_car_racing_patch.sh` from `ReQueST/scripts`
4.  Replace `gym/envs/box2d/car_racing.py` with `ReQueST/scripts/car_racing.py`
5.  Clone the
    [world models repo](https://github.com/hardmaru/WorldModelsExperiments/)
6.  Download
    [MNIST](https://github.com/lucastheis/deepbelief/blob/master/data/mnist.npz)
7.  Set `wm_dir`, `mnist_dir`, and `home_dir` in
    `ReQueST/utils.py`
8.  Install the `rqst` package with `python setup.py install`
9.  Download
    [data.zip](https://drive.google.com/file/d/1jeNCyhN7LB4TXmDP8ePnxBeGI3eLyi3m/view?usp=sharing),
    then unzip it into `ReQueST/data`
10. Jupyter notebooks in `ReQueST/notebooks` provide an entry-point to the code base, where you can
    play around with the environments, visualize synthesized queries, and reproduce the figures
    from the paper.

## Citation

If you find this software useful in your work, we kindly request that you cite the following
[paper](https://arxiv.org/abs/1912.05652):

```
@article{ReQueST,
  title={Learning Human Objectives by Evaluating Hypothetical Behavior},
  author={Reddy, Siddharth and Dragan, Anca D. and Levine, Sergey and Legg, Shane and Leike, Jan},
  journal={arXiv preprint arXiv:1912.05652},
  year={2019}
}
```

## Disclaimer

This is not an officially supported Google product.
