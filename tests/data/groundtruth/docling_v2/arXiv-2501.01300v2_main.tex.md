1
.001

Thermodynamics of charmed hadrons across chiral crossover from lattice QCD    
S.Sharma et al.

# [

mode = title]Thermodynamics of charmed hadrons across chiral crossover from lattice QCD

1]Sipaz Sharma[orcid=0000-0001-6916-2233]
sipaz.sharma@tum.de
[1]organization=Physik Department, Technische Universitat Munchen,
 addressline=James-Franck-Strae 1, 
 city=Garchingb.Munchen,
 postcode=D-85748, 
 country=Germany

2]Frithjof Karsch
        [2]organization=Fakultat fur Physik, Universitat Bielefeld,
 	addressline=Universitätsstraße 25, 
 	city=Bielefeld,
 	postcode=D-33615, 
 	country=Germany

3]Peter Petreczky
        [3]organization=Physics Department, Brookhaven National Laboratory,
 	city=Upton, NY,
 	postcode=11973, 
 	country=USA

## Abstract

We use up to fourth-order charm fluctuations and their correlations with net baryon number, electric charge, and strangeness fluctuations, calculated within the framework of lattice QCD, to study the continuum partial pressure contributions of charmed baryons and mesons. We show that, at and below the chiral crossover temperature, these partial pressures receive enhanced contributions from experimentally unobserved charmed hadrons predicted by the Quark Model. Additionally, we demonstrate that at the chiral crossover, the Hadron Resonance Gas description breaks down, signaling the modification of open charm hadrons and thereby implying the onset of charm deconfinement. We present evidence for the survival of low-lying non-radial $1S$ and $1P$ hadron-like excitations above the chiral crossover, which hints at the sequential melting of charmed hadrons. Finally, we investigate the continuum partial pressure contribution of charm quark-like excitation that emerges at the chiral crossover and calculate its temperature-dependent in-medium mass.

\_set:nn stm / mktitle nologo

## Introduction

Lattice studies in the recent years have predicted the existence of experimentally unobserved charmed resonances by calculating charm fluctuations and their correlations with other conserved quantum numbers a.k.a. generalized charm susceptibilities [BAZAVOV2014210, Sharma:2022ztl, Bazavov:2023xzm]. Recently, analysis of experimental  open charm hadron yields using the extended Statistical Hadronization Model for charm (SHMc) [ANDRONIC2019759], showed enhanced yields in the charmed baryon sector [Braun-Munzinger:2024ybd]. This enhancement is relative to the experimentally known charmed hadronic states tabulated in the Particle Data Group (PDG) records, and  therefore corroborates the existence of missing resonances in the open charm sector.

Until now, lattice QCD predictions of missing resonances have been based solely on the ratios of generalized charm susceptibilities. This approach, however, could not provide a quantitative measure of enhancement in individual quantum number channels. The reason lattice QCD studies had to resort to ratios of generalized susceptibilities rather then the susceptibilities themselves is because
controlling the cutoff effects in the latter is challenging due to the relatively large charm quark mass.  In this study, for the first time, we use our continuum extrapolated lattice QCD results to explicitly determine enhancement factors at the chiral crossover temperature,

${T\_{pc}=(156.5\pm1.5)}$ MeV [HotQCD:2018pds, Borsanyi:2020fev], for both the charmed baryonic and mesonic sectors.

The chiral crossover also exhibits deconfining feature in the charm quark sector, i.e. charm quarks as new degrees of freedom start appearing at or close to the pseudo-critical temperature of chiral symmetry restoration (Bazavov et al. (2024)). Clearly the appearance of charm quark degrees of freedom at

$T\_{pc}$, arising due to the disappearance of some hadronic states, will also liberate light quark degrees of freedom, although in the chiral limit this may not directly be related to the universal aspects of the chiral phase transition. The partial charm hadronic pressures start to deviate from the HRG model prediction, but not all charmed
hadrons dissociate at $T\_{pc}$. Furthermore, recent lattice QCD studies provide compelling evidence for the survival of charmed hadrons above the chiral crossover temperature [Bazavov:2023xzm, Sharma:2024ucs, Sharma:2024edf]. In this paper, we investigate the gradual deconfinement of charmed hadrons by analyzing continuum extrapolated proxies for partial pressures of charmed baryons and mesons across the chiral crossover.

## Calculating generalized susceptibilities on the lattice

To project out the relevant degrees of freedom in the charm sector, one calculates the generalized susceptibilities, ${\chi^{BQSC}\_{klmn}}$, of the conserved charges: baryon number, $B$, electric charge, $Q$, strangeness, $S$, and charm, $C$. This involves taking appropriate derivatives of the total pressure,

$$P=\frac{T}{V}\ln Z(T,V,\hat{\mu}_B,\hat{\mu}_Q,\hat{\mu}_S,\hat{\mu}_C)
    \; ,$$

which contains contributions from the total charm pressure, $P\_C$. These derivatives are taken with respect to the chemical potentials, ${\hat{\mu}\_X = \mu\_X/T}$, ${\forall X \in \{B, Q, S, C\}}$, of the quantum number combinations one is interested in,

$${\chi^{BQSC}_{klmn}=\dfrac{\partial^{(k+l+m+n)}\;\;[P\;(\hat{\mu}_B,\hat{\mu}_Q,\hat{\mu}_S,\hat{\mu}_C)\;/T^4]}{\partial\hat{\mu}^{k}_B\;\;\partial\hat{\mu}^{l}_Q\;\;\partial\hat{\mu}^{m}_S\;\;\partial\hat{\mu}^{n}_C}}\bigg|_{\overrightarrow{\mu}=0}\text{,}$$

where $\vec{\mu}=(\mu\_B, \mu\_Q, \mu\_S, \mu\_C)$. 
Note that ${\chi^{BQSC}\_{klmn}}$ will be non-zero only for ${(k+l+m+n) \in \text{even}}$. In the following, if the subscript corresponding to a conserved charge is zero in the
left hand side of Eq. [eq:chi], then both the corresponding superscript as well the zero subscript will be suppressed.

Generalized susceptibilities up to fourth order have been calculated on the (2+1)-flavor HISQ (Highly Improved Staggered Quark) gauge configurations generated by the HotQCD collaboration

[Bollweg:2021vqf] for the physical strange to light quark mass ratio, ${m\_s/m\_{l}}=27$. The temperature scale was set using the $f\_K$ scale [Bollweg:2021vqf]. 
The charm-quark sector has been treated in the quenched approximation. We use HISQ action with epsilon term for charm quark to remove $\mathcal{O}((am\_c)^4)$ tree-level lattice artifacts [Follana:2006rc]. This setup is known to have small discretization effects for charm quark related observables [Follana:2006rc,MILC:2010pul]. The calculation of $\chi^{BQSC}\_{klmn}$ involves derivatives of the pressure and on the lattice this is achieved by the unbiased stochastic estimation of various traces  consisting of inversions and derivatives  of the fermion matrices ($D$)  using the random noise method [Mitra:2022vtf].

Lattice cutoff effects cancel to a large extent in the ratios of various generalized susceptibilities

[Bazavov:2023xzm]. These ratios, even though calculated on lattices with a finite lattice spacing, $a$, are close to
the continuum limit results.  Therefore, in this work we use generalized susceptibilities normalized by $\chi^C\_4$ calculated on lattices with a fixed temporal extent, $N\_\tau=8$, where $N\_\tau=(aT)^{-1}$.   In order to understand the cut-off effects due to heavier charm quark, we used two different Lines of Constant Physics (LCPs) to tune the bare charm quark mass $am\_c$.  The first LCP (LCP$\_{[a]}$) corresponds to keeping the spin-averaged charmonium mass, ${(3m\_{J/\psi}+m\_{\eta\_{c\bar{c}}})/4}$, fixed to its physical value [Sharma:2022ztl]. The second LCP (LCP$\_{[b]}$)  is defined by the physical (PDG) charm to strange quark mass ratio, $m\_c/m\_s=11.76$ [ParticleDataGroup:2022pth].  Further details of the charm-quark mass tuning and parametrization can be found in [Sharma:2024ucs]. Here we use ratios of generalized susceptibilities calculated on lattices with temporal extent $N\_\tau=8$ on LCP$\_{[b]}$ because of their relatively smaller errors.  To convert generalized susceptibilities normalized by $\chi^C\_4$ into absolute numbers we use continuum extrapolated results for $\chi^C\_4$. Details of the continuum extrapolation will be given in a forthcoming  publication.

## Boltzmann Approximation

Image: BQSC-1003.pdf

<!-- image -->

Image: D-BQSC-0004-1003.pdf

<!-- image -->

Comparison of QM-HRG and PDG-HRG predictions for charmed baryons [Left] and charmed mesons [Right]. $\Delta=100 |1-\text{QM-HRG}/\text{PDG-HRG}|\_{T\_{pc}}$. The yellow bands represent $T\_{pc}$ with its uncertainty.

### Charmed Hadrons

Hadron resonance gas model (HRG) has been successful in describing the particle abundance ratios measured in the heavy-ion experiments. It describes a non-interacting gas of hadron resonances, and therefore can be used to calculate the hadronic pressure below ${T\_{pc}}$ [ANDRONIC2019759]. In the Boltzmann approximation, the dimensionless partial pressures from the charmed meson, ${P^C\_{M}},$  and the charmed baryon, ${P^C\_{B}}$, sectors take the following forms [Allton:2005gk]:

$$\begin{gather}
	\begin{aligned}
		{P^C_{M}(T,\overrightarrow{\mu})}&{=\dfrac{1}{2\pi^2}\sum_{i\in \text{C-mesons}}g_i \bigg(\dfrac{m_i}{T}\bigg)^2K_2(m_i/T)}\\
			&{\text{cosh}(Q_i\hat{\mu}_Q+S_i\hat{\mu}_S+C_i\hat{\mu}_C)} \text{ ,}\\
		{P^C_{B}(T,\overrightarrow{\mu})}&={\dfrac{1}{2\pi^2}\sum_{i\in \text{C-baryons}}g_i \bigg(\dfrac{m_i}{T}\bigg)^2K_2(m_i/T)}\\
			&{\text{cosh}(B_i\hat{\mu}_B+Q_i\hat{\mu}_Q+S_i\hat{\mu}_S+C_i\hat{\mu}_C)} \text{ .}
		
	\end{aligned}
\end{gather}$$

In above equations, at a given temperature, the summation is over all charmed mesons/baryons (C-mesons/baryons) with masses given by

${m\_i}$; degeneracy factors of the states with equal mass and same quantum numbers are represented by ${g\_i}$; ${K\_2(x)}$ is a modified Bessel function, which for a large argument can be approximated by
${K\_2(x)}\sim\sqrt{\pi/2x}\;e^{-x}\;[1+\mathbb{O}(x^{-1})]$. Consequently, if a charmed state under consideration is much heavier than the relevant temperature scale, such that ${m\_i\gg T}$, then the contribution to ${P\_C}$ from that particular state will be exponentially suppressed, e.g., the singly-charmed
${\Lambda}\_c^{+}$ baryon has a PDG mass of about $2286$MeV, whereas the doubly-charmed ${\Xi\_{cc}^{++}}$ baryon's mass as tabulated in PDG records is about $3621$ MeV, hence at ${T\_{pc}}$, the contribution to ${P^C\_{B}}$ from ${\Xi\_{cc}^{++}}$ will be suppressed by a factor of $10^{-4}$ in relation to ${\Lambda}\_c^{+}$ contribution. Therefore, in the validity regime of Boltzmann approximation, substituting Eq. [eq:McBc] in Eq. [eq:chi] implies $P\_C(T,\vec{\mu})\approx\chi\_2^C\approx\chi\_n^C$, for $n$even.

Charm fluctuations calculated in the framework of lattice QCD can receive enhanced contributions due the existence of not-yet-discovered open-charm states. It is possible to compare this enhancement to the HRG calculations performed with two data sets. The first scenario, denoted by PDG-HRG, is based on the experimentally established states tabulated in the PDG records. The second scenario, denoted by QM-HRG, in addition to PDG states, takes into account states predicted via Quark-Model calculations

[Ebert:2009ua, Ebert:2011kk, Chen:2022asf]. In the validity regime of HRG, $\chi^{nm}\_{BC}\approx P^C\_{B}$, and $P^C\_{M} = P\_C-P^C\_{B}$. Fig. [fig:barCmesC] [Left] shows that the QM-HRG predicts a $92\%$ enhancement of the partial charmed baryon pressure at $T\_{pc}$ compared to the PDG-HRG expectation, whereas the predicted enhancement in the partial charmed meson pressure is only $15\%$. These results suggest that, based on the QM-HRG predictions, the charmed baryonic sector is significantly more incomplete than the charmed mesonic sector. In recent years, the 
 experimental
 discovery of various $\Lambda\_C$ and $\Omega\_C$ resonances, in particular by LHCb and Belle collaborations [Belle:2016tai,LHCb:2017uwr,LHCb:2017jym,Chen:2017gnu] have confirmed this.

### Charm Quarks

Charm quarks offer an advantage over the light quarks because for temperatures a few times $T\_{pc}$, the Boltzmann approximation works for an ideal massive quark-antiquark gas [Allton:2005gk,BAZAVOV2014210]. Therefore, in this approximation, the dimensionless partial charm quark pressure, $P^C\_{q}$, is given by,

$$P^C_{q}(T,\overrightarrow{\mu})&=&\dfrac{3}{\pi^2}\bigg(\dfrac{m_q^C}{T}\bigg)^2K_2(m_q^C/T)\cdot \nonumber \\
	&&\text{cosh}\bigg(\dfrac{2}{3}\hat{\mu}_Q+\dfrac{1}{3}\hat{\mu}_B+\hat{\mu}_C\bigg) \text { ,}$$

where $m\_q^C$is the pole mass of the charm quark, and the degeneracy factor is 6.

## Enhancement of charmed hadrons

Image: P\_B.pdf

<!-- image -->

Image: P\_M.pdf

<!-- image -->

Shown are partial charmed baryon pressure [Left] and partial charmed meson pressure [Right] along with the respective QM-HRG, PDG-HRG and 1S1P-HRG (see text) predictions. The yellow bands represent $T\_{pc}$ with its uncertainty.

To investigate the nature of charm degrees of freedom above  $T\_{pc}$, we extend the simple hadron gas model allowing the presence of partial charm quark pressure based on Ref.[Mukherjee:2015mxc],

$$\begin{align}
	P_C(T,\vec{\mu})=P^C_{M}(T,\vec{\mu})+P^C_{B}(T,\vec{\mu})+P^C_{q}(T,\vec{\mu}) \, .
	
\end{align}$$

In our recent works [Bazavov:2023xzm, Sharma:2024ucs, Sharma:2024edf], we show that this model successfully passes numerous validity tests and satisfies various constraints [Bazavov:2023xzm]. In this quasi-particle model, the partial pressures of quark-, baryon- and meson-like excitations for $\vec{\mu}=0$can be expressed in terms of the generalized susceptibilities as follows,

$$\begin{align}
	P^C_{q}&=9(\chi^{BC}_{13}-\chi^{BC}_{22})/2\; , 
	 	\\
	P^C_{B}&=(3\chi^{BC}_{22}-\chi^{BC}_{13})/2\; , 
	\\
	P^C_{M}&=\chi^{C}_{4}+3\chi^{BC}_{22}-4\chi^{BC}_{13} \; .
	
\end{align}$$

Note that in the validity regime of HRG, $\chi^{BC}\_{22}\approx\chi^{BC}\_{13}$. Hence, in this phase the partial pressure contribution from quark-like excitations, $P^C\_q$ is, by construction, zero, and $P^C\_B$ and $P^C\_M$ reduce to $\chi^{BC}\_{13}$ and $\chi^4\_C-\chi^{BC}\_{13}$, respectively.

Continuum estimates for charmed baryon and meson partial pressures shown in Fig.

[fig:enhan] are obtained by multiplying continuum extrapolated $\chi^C\_4$ values to ratios $P^C\_B/P\_C$ and $P^C\_M/P\_C$  shown in our previous work [Bazavov:2023xzm]. As mentioned earlier, these ratios, despite being calculated on a finite temporal lattice extent, are largely cut-off independent. For $T&lt;T\_{pc}$, Fig. [fig:enhan] shows a clear agreement of the QM-HRG predictions with the  partial hadron pressures calculated on the lattice. This agreement, in addition to corroborating  the validity of HRG in the low temperature phase, confirms the existence of experimentally unobserved hadronic states. In particular, at the chiral crossover, in the charmed baryonic sector, $P^C\_B$ calculated on the lattice is almost twice as large as the PDG-HRG expectation, whereas $P^C\_M$ calculated on the lattice is around $20\%$ larger than the  PDG-HRG prediction. The enhancement factors explicitly quoted in Fig. [fig:enhan] agree within errors with the enhancement predictions quoted in Fig. [fig:barCmesC].

## Charm degrees of freedom in QGP

Image: P\_q.pdf

<!-- image -->

Image: m\_q.pdf

<!-- image -->

Shown is the partial pressure contribution of charm quark-like excitation above $T\_{pc}$ [Left] and the temperature dependent in-medium mass of a charm quark-like excitation above $T\_{pc}$ [Right]. The yellow bands represent $T\_{pc}$ with its uncertainty.

Fig. [fig:Pq] [Left] shows that the partial pressure contribution to $P\_C$ from charm quark-like excitation becomes non-zero at $T\_{pc}$, and starts increasing as a function of temperature. This increasing $P^C\_q$ results in a decreasing relative partial pressure contribution to $P\_C$ from hadron-like excitations. At the highest temperature shown in Figs. [fig:enhan] and [fig:Pq] [Left], i.e., $175$ MeV, the relative partial pressure contributions from hadron and quark-like excitations cross. Based on this trend, it is expected that above $T\sim175$ MeV, $P^C\_q$ will become the dominant contribution. Therefore, eventually $P^C\_B$ and $P^C\_M$ shown in Fig. [fig:enhan]will turn around and start decreasing. At sufficiently high  temperatures partial pressure contribution from charmed hadron-like excitations will go to zero.

Departure of charmed hadronic pressures from their respective QM-HRG predictions hints at a sequential melting pattern. This means that the higher excited charmed hadrons start dissociating at

$T\_{pc}$, whereas the ground state charmed hadrons survive inside the QGP. To put this on more solid grounds, we performed another HRG model calculation based on the low-lying, $1S$ and $1P$, charmed hadrons tabulated in [Chen:2022asf]  we label this as 1S1P-HRG. For $T\_{pc}&lt;T\leq 166.1$MeV, both $P^C\_B$ and $P^C\_M$ in Fig. [fig:enhan] are described by 1S1P-HRG. For $P^C\_M$, both PDG-HRG and 1S1P-HRG almost overlap, and only at the highest temperatures shown in Fig. [fig:enhan] [Right] there are some visible differences. This is expected because all low-lying charmed meson states are experimentally known. In addition to low-lying 1S and 1P states, PDG-HRG contains one charmed-light meson: $D\_3^*(2750)$ and two charmed-strange mesons: $D\_{s1}^*(2700)$ and $D\_{s3}^*(2860)$. We used PDG masses of $D$, $D^*$, $D^*\_0$ and $D\_1$ charmed-light mesons. If chiral partners were to get degenerate at $T\_{pc}$, then the scalar and axial-vector mesons would get lighter, 
making 1S1P-HRG curve overshoot $P^C\_M$ above $T\_{pc}$. This indicates that the charm sector is not strongly influenced 
by the chiral symmetry restoration. This is also in line with studies of screening masses for charmed parity partners, which indicate that unlike in the light quark sector the charmed screening masses  become almost degenerate only at temperatures above $2T\_{pc}$ [Bazavov:2014cta]. Similar observation holds for the charmed-strange mesonic sector. In the charmed baryonic sector, after departure from 1S1P-HRG, for the highest two temperatures, PDG-HRG prediction can very well describe $P^C\_B$ in Fig. [fig:enhan] [Left]. The fact that 1S1P-HRG lies above PDG-HRG for $P^C\_B$indicates that there are many low-lying and therefore thermodynamically dominant charmed baryons missing from the PDG record.

In the coexistence phase of charmed hadron and quark-like excitations,

$m^C\_q$ in Eq. [eq:Qc] becomes temperature dependent and can be interpreted as the mass of a quasi-particle with quantum numbers of charm quark. Given $P^C\_q(T)$ as a function of temperature, one can solve Eq. [eq:Qc] at each temperature and obtain temperature dependence of $m^C\_q$. In Fig. [fig:Pq] [Right], at each temperature, the error on $m^C\_q$ is the standard deviation of $m^C\_q$ values obtained after solving Eq.[eq:Qc] for $50$ fake Gaussian samples. Fig. [fig:Pq] [Right] shows that at $T=162.4$ MeV, $m^C\_q$ is around the mass of D-meson, and starts decreasing with temperature. Quasi-particle model in Eq. [eq:Pmodel] considers a non-interacting gas of charmed hadron and quark-like excitations, but the temperature dependence of $m^C\_q$ encodes the in-medium interactions. In the non-interacting quark gas limit, $m^C\_q$will become the pole mass of charm quark.

## Conclusions

In this work, for the first time, we showed that in the low temperature phase HRG calculations agree with various continuum extrapolated generalized susceptibilities of charm calculated in the framework of lattice QCD when adding
experimentally unobserved charmed hadrons, predicted by the Quark Model calculations, to the HRG model spectrum. Our previous studies, predicting the existence of missing charmed resonances, utilised ratios of various generalized susceptibilities. We showed that continuum extrapolated partial pressures of charmed baryons and mesons calculated on the lattice receive enhanced contributions from these missing resonances. We concluded that the charmed baryon sector is significantly more incomplete than the charmed meson sector.

We showed that above chiral crossover, the charm pressure can be decomposed into partial pressures of charm quark, charmed mesons, and charmed baryon-like excitations. At the highest temperature,

$T \sim 176$ MeV, studied by us, the charm pressure receives equal contributions from charmed hadrons and charm quark, indicating that charmed hadrons can exist in Quark Gluon Plasma (QGP).

## Acknowledgments

This work was supported by The Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - Project number 315477589-TRR 211,
”Strong interaction matter under extreme conditions”. This material is based upon work supported by The U.S. Department of Energy, Office of Science, Office of Nuclear Physics through Contract No.

DE-SC0012704 and the Topical Collaboration in Nuclear Theory Heavy-Flavor Theory (HEFTY) for QCD Matter.
The authors gratefully acknowledge the
computing time and support provided to them on the high-performance computer Noctua 2 at the NHR Center
PC2 under the project name: hpc-prf-cfpd. These are funded by the Federal Ministry of Education
and Research and the state governments participating on the basis of the resolutions of the GWK
for the national high-performance computing at universities (www.nhr-verein.de/unsere-partner).
Numerical calculations have also been performed on the
GPU-cluster at Bielefeld University, Germany. We thank the Bielefeld HPC.NRW team for their support.

All computations in this work were performed using

SIMULATeQCD code [HotQCD:2023ghu]. All the HRG calculations were performed using the AnalysisToolbox code developed by the HotQCD Collaboration [Clarke:2023sfy].

cas-model2-names

refs