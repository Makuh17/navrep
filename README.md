## NavRep
This repo forks the original NavRep repo mainly developed by Daniel Dugas:
https://github.com/ethz-asl/navrep

## This work
This work explores three avenues aiming to located ways of improving navigation success rate in realistic crowded environments. The three avenues are:

- Pretrained initialisation based on behaviour cloning of an expert policy, [`pretrain`](https://github.com/Makuh17/navrep/tree/pretrain)
- Intrinsic curiosity driven learning, [`reward`](https://github.com/Makuh17/navrep/tree/reward)
- Curriculum based learning using more realistic maps, [`curriculum`](https://github.com/Makuh17/navrep/tree/curriculum)

The approaches are explored independently on different branches, linked above. The individual READMEs explain how to use the scripts.

## Credits

This library was written primarily by Daniel Dugas. The transformer block codes, and vae/lstm code were taken or heavily derived from world models and karpathy's mingpt. We've retained the copyright headers for the relevant files.
