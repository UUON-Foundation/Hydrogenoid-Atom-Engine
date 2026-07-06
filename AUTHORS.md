# Authors and Credit

| Date       | Contributor               | What they did                                        |
|------------|----------------------------|-------------------------------------------------------|
| 2026-07-06 | Phillip Aguilar Ruiz III   | Founder, Principal Investigator, UUON Foundation Inc. Directed the project and reviewed the work. |
| 2026-07-06 | Claude (Anthropic)         | Built the wavefunction math, the sampling and isosurface method, the rendering, and the interaction design. |

## Notes on credit

The core physics (the Laguerre and Legendre math, the hydrogen atom
wavefunctions) is standard, publicly known quantum mechanics. It is not a
UUON Foundation invention, and no proprietary UUON framework is used
anywhere in this engine.

What is original here is the engineering: the specific way the point cloud
is sampled, and the shortcut used to compute the outer surface quickly. Those
choices are explained in the README and in comments in the code.
