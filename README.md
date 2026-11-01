# Surgical Tattoos for Knee Arthroplasty

**Topic area:** Computer vision, fiducial markers

## Problem

Robotic-assisted knee surgeries (e.g. MAKO) are more precise than manual surgeries, but require dynamic **tracking of the leg bones via optical markers** which are pinned into the bones. This is necessary for the robotic assistance, but harms the bone.

## Background and motivation

Less invasive would be **visual markers that are applied to the skin**, such as tattoos. However, soft tissue displacement occurs here, and the positions of the markers relative to the bone are not rigid. This makes it tricky to estimate the bone axes.

## Goal

Experimental evaluation of a non-invasive skin-based tracking method for tibia and femur axes.

## Proposed solution

Explore and compare skin-based marker approaches to ground truth data (e.g. using OptiTrack).

## Tasks and objectives

Using an artificial knee, create a prototype which compares skin-based marker precision to ground truth femur and tibia axes.

## References

- Keller, M., Werling, K., Shin, S., Delp, S., Pujades, S., Liu, C. K., & Black, M. J. (2023). From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans. *ACM Transactions on Graphics*, 42(6), Article 553. https://doi.org/10.1145/3618381. Code: https://github.com/MarilynKeller/SKEL
- Coburn, K., Velhal, M., Goenezen, S., & Liang, H. (2023). Digital image correlation through surface patterning of skins to monitor knee motion. *Biotribology*, 35, 100251.
- Zamora, R., Punt, S. E., Christman-Skieller, C., Yildirim, C., Shapton, J. C., & Conrad 3rd, E. U. (2019). Are skin fiducials comparable to bone fiducials for registration when planning navigation-assisted musculoskeletal tumor resections in a cadaveric simulated tumor model? *Clinical Orthopaedics and Related Research*, 477(12), 2692.
- Tsukada, S., Ogawa, H., Nishino, M., Kurosaka, K., & Hirasawa, N. (2019). Augmented reality-based navigation system applied to tibial bone resection in total knee arthroplasty. *Journal of Experimental Orthopaedics*, 6(1), 44.
- Tsukada, S., Ogawa, H., Nishino, M., Kurosaka, K., & Hirasawa, N. (2021). Augmented reality-assisted femoral bone resection in total knee arthroplasty. *JBJS Open Access*, 6(3), e21-00001.
- Ancillao, A., Aertbeliën, E., & De Schutter, J. (2021). Effect of the soft tissue artifact on marker measurements and on the calculation of the helical axis of the knee during a gait cycle: A study on the CAMS-Knee data set. *Human Movement Science*, 80, 102866.
- Rath, B., Huber, T., Kamper, M., Beckmann, J., Hirschmann, M. T., Hammer, N., & Ullmann, D. (2026). Feasibility and accuracy of intraoperative 3D surface scanning to assess femoral component alignment relative to the native distal femoral articular surface in total knee arthroplasty. *Knee Surgery, Sports Traumatology, Arthroscopy*.
- Agarwal, B. B. (2009). Tattoo: a navigational tool, a lighthouse for biomolecular level precision in surgical endoscopy. Newton's apple or Eve's? *Surgical Endoscopy*, 23(5), 1166-1167.