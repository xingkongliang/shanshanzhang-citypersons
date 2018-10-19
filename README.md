# README #

This repo provides bounding box annotations, python evaluation code, and a benchmark for CityPersons, which is a subset of the [Cityscapes dataset](https://www.cityscapes-dataset.com/).
Please download the images from the Cityscapes website!

Welcome to join the competition by submitting your results on the test set!

### Benchmark ###
|         Method         | MR (Reasonable) | MR (Reasonable_small) | MR (Reasonable_occ=heavy) | MR (All) |
|:----------------------:|:---------------:|:---------------------:|:-------------------------:|:--------:|
| [Repultion Loss](http://arxiv.org/abs/1711.07752)     |      11.48%     |         15.67%        |           52.59%          |  39.17%  |
|  [Adapted FasterRCNN](http://202.119.95.70/cache/12/03/openaccess.thecvf.com/f36bf52f1783160552c75ae3cd300e84/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf)  |      12.97%     |         37.24%        |           50.47%          |  43.86%  |

[comment]: <![leaderboard.png](https://bitbucket.org/repo/XXegAKG/images/1374766803-leaderboard.png)> 

Please refer to the [instructions](https://bitbucket.org/shanshanzhang/citypersons/src/f44d4e585d51d0c3fd7992c8fb913515b26d4b5a/evaluation/?at=default) on submitting results for evaluation.

### What Do We Have? ###

* [Train/val annotations](https://bitbucket.org/shanshanzhang/citypersons/src/f44d4e585d51d0c3fd7992c8fb913515b26d4b5a/annotations/?at=default)
* [Python evaluation code](https://bitbucket.org/shanshanzhang/citypersons/src/f44d4e585d51d0c3fd7992c8fb913515b26d4b5a/evaluation/eval_script/?at=default)
* Competition leaderboard for the test set


### Annotation Example ###
![图片1.png](https://bitbucket.org/repo/XXegAKG/images/982984467-%E5%9B%BE%E7%89%871.png)

### Citation ###

If you use this data and code, please kindly cite the following papers:


```
#!bibtex
@INPROCEEDINGS{Shanshan2017CVPR,

  Author = {Shanshan Zhang and Rodrigo Benenson and Bernt Schiele},

  Title = {CityPersons: A Diverse Dataset for Pedestrian Detection},

  Booktitle = {CVPR},

  Year = {2017}
 }

@INPROCEEDINGS{Cordts2016Cityscapes,

title={The Cityscapes Dataset for Semantic Urban Scene Understanding},

author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},

booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},

year={2016}
}

```
---------------------------------------------------------------------------------------------------------------------
This material is presented to ensure timely dissemination of scholarly and technical work. Copyright and all rights therein are retained by authors or by other copyright holders. All persons copying this information are expected to adhere to the terms and constraints invoked by each author’s copyright. In most cases, these works may not be reposted without the explicit permission of the copyright holder.