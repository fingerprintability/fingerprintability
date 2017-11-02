# fingerprintability
Source code for the data collection and analysis used in the 'How unique is your onion?' project.


## Data collection
We used [tor-browser-selenium](https://github.com/webfp/tor-browser-selenium) and [tor-browser-crawler](https://github.com/webfp/tor-browser-crawler) libraries to collect data from the Onion services:
https://github.com/webfp

## Credits
We used code from the following papers/projects. We thank respective authors for being kind to share their code:

* Panchenko et al., "Website Fingerprinting at Internet Scale" in NDSS
2016: http://lorre.uni.lu/%7Eandriy/zwiebelfreunde/

* Wang et al., "Effective Attacks and Provable Defenses for Website
Fingerprinting": https://www.cse.ust.hk/~taow/wf/

* Tobias Pulls implementation of Wang et al's USENIX'14 kNN-based
attack: https://github.com/pylls/go-knn

* Hayes and Danezis, "k-fingerprinting: a Robust Scalable Website
Fingerprinting Technique" in USENIX 2016: https://github.com/jhayes14/k-FP

* Acknowledgments to George Danezis for the source code of a web traffic
parser.
