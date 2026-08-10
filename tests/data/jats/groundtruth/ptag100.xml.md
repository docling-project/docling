# Time-Dependent 3D Oscillator with Coulomb Interaction: An Alternative Approach for Analyzing Quark–Antiquark Systems

Jeong Ryeol Choi, Salim Medjber, Salah Menouar, Ramazan Sever

## Abstract

In this work, the dynamics of quark–antiquark pair systems is investigated by modeling them as general time-dependent 3D oscillators perturbed by a Coulomb potential. Solving this model enables the prediction of key mesonic properties such as the probability density, energy spectra, and quadrature uncertainties, offering theoretical insights into the confinement of quarks via gluon-mediated strong interactions. To tackle the mathematical difficulty raised by the time dependence of parameters in the system, special mathematical techniques, such as the invariant operator method, unitary transformation method, and the Nikiforov–Uvarov functional analysis are used. The wave functions of the system, derived using these mathematical techniques, are expressed analytically in terms of the Gauss hypergeometric function whose mathematical properties are well characterized. Our results provide the quantum-mechanical framework of quark–antiquark systems that are essential for exploring the nonperturbative aspects of quantum chromodynamics. In addition, the underlying mathematical structure may serve as a foundation for addressing broader challenges in particle physics, including the origin of mass and its connection to the Higgs mechanism.

## 1.

Quark–antiquark pairs are a key concept in particle physics, essential for understanding the strong interaction and the structure of hadrons such as mesons and baryons. Typical examples of quark–antiquark pairs are bound states of a heavy quark and its antiquark, such as charmonium and bottomonium, collectively referred to as quarkonia [1]. Quarkonia can be readily produced in experiments such as the LHC, and their properties can be measured with high precision. Quarkonia’s distinct mass spectra, unique decay patterns, and relatively long lifetimes are of significant interest in the search for physics that extends past the Standard Model. As such, they may offer a promising avenue for probing potential new phenomena not accounted for by the Higgs sector [2,3]. This makes quarkonia an important topic requiring collaborative research between experimental and theoretical physicists. Indeed, they are actively studied in connection with new particle searches and dark matter theories [4–7].

In light of this, it is essential to establish a theoretical framework that enables systematic analysis and interpretation of quarkonium systems. In particle physics, quark–antiquark systems are commonly modeled using a potential that combines harmonic and Coulomb terms [8–10], where the harmonic component captures the long-range confinement effect and the Coulomb term accounts for the short-range interactions based on quantum chromodynamics (QCD). Motivated by this, we focus on the analysis of time-dependent 3D oscillators perturbed by a Coulomb potential throughout this work. The quantum-mechanical solutions of such systems are known to be expressible in terms of biconfluent Heun functions [11,12]. However, limited knowledge of the properties of such functions continues to make these systems difficult to analyze. At present, the characterization of these functions remains confined to their behavior in certain special cases and their connections to specific recursion relations, particularly their representations in terms of these relations [13,14]. This lack of comprehensive understanding is the reason why many researchers refer to them primarily through their defining equation, the biconfluent Heun equation, rather than treating them as fully characterized functions in their own right [15–19].

To overcome this limitation, the present work explores an alternative analytical treatment by expressing the solutions in terms of well-established mathematical functions, thereby providing a more accessible and rigorous theoretical understanding of the system. Deriving such solutions to fundamental equations for our general physical systems governed by time-dependent Hamiltonians [20–22] may require special mathematical techniques beyond conventional methods such as the separation of variables. One promising approach is the invariant operator method [23–26], which has proven effective in the study of time-dependent Hamiltonian systems (TDHSs). This method allows for the construction of exact quantum solutions in systems with time-dependent parameters. Specifically, if an invariant operator can be identified for a given system, the quantum wave functions, apart from phase parts, can be expressed by eigenfunctions of this operator. Consequently, solving the quantum system reduces to finding the eigenfunctions of the invariant operator, while the required phase factors can be determined with the help of the Schrödinger equation.

Regarding coupling of potentials, as well as the system’s time dependence, the general form of the invariant operator is typically complicated. Therefore, to solve its eigenvalue equation, more advanced mathematical techniques, such as the unitary transformation method [27,28] and the Nikiforov–Uvarov functional analysis (NUFA) [29], are necessary. The NUFA method builds upon and integrates several approaches, including the original Nikiforov–Uvarov (NU) method [30], the parametric NU method [31–34], and elements of functional analysis [35]. This enhanced method enables one to obtain the solution of related equations, including those of hypergeometric type, in a systematic and elegant manner, allowing the determination of the eigenvalue spectrum and the corresponding wave functions. We demonstrate that the wave functions derived through this approach can be expressed in terms of the well-known Gauss hypergeometric function. This may provide a significantly more tractable framework for analyzing the system compared to existing solutions that are based on biconfluent Heun functions. Taking advantage of these strengths, we show that the newly obtained solutions enable a thorough investigation of the system, particularly in dynamically evolving hadronic environments where effective quark interactions may vary due to medium-induced effects and nonequilibrium processes.

The main contributions of the present work can be summarized as follows. First, we develop an analytical treatment for a time-dependent quark–antiquark model based on a 3D oscillator perturbed by a Coulomb interaction within the framework of invariant operator theory. Second, by combining the invariant operator method, unitary transformation techniques, and the NUFA formalism, we derive analytical expressions for the eigenspectrum, quantum phases, and wave functions of the system in closed form. In contrast to several previous approaches in which the solutions are represented in terms of biconfluent Heun functions, the present treatment expresses the wave functions in terms of the Gauss hypergeometric function, whose mathematical properties are more thoroughly established. This provides a more tractable analytical framework for investigating the dynamical properties of quark–antiquark systems governed by time-dependent Hamiltonians.

## 2.

### 2.1.

#### 2.1.1.

We consider a quark–antiquark pair with a time-dependent effective reduced mass $\mu (t)$ in spherical coordinates, governed by a central potential $g(t)r^{2}-\frac{Z(t)}{r}$ where $g(t)$ and $Z(t)$ are coefficients that depend on time. While $g(t)r^{2}$ represents a confining trap potential with adjustable strength, $-\frac{Z(t)}{r}$ is a Coulomb perturbation with the constraint $Z(t)\gt 0$ . As the distance between quarks increases, the harmonic potential rises sharply, effectively prohibiting their free separation within a bound state. This potential acts as a simplified representation of the QCD confinement mechanism and dominates at large separations. Meanwhile, the Coulomb term represents the electromagnetic or color interaction mediated by the color charge of one quark acting on the other, effectively serving as a short-range attractive force between the quark and antiquark.

Time-dependent Hamiltonians naturally arise in hadron physics when quark–antiquark systems are embedded in dynamical environments, such as quark–gluon plasma or nonequilibrium hadronic media produced in high-energy collisions [20–22]. In such situations, the effective interaction between quarks may vary with time due to medium evolution, color-screening effects, confinement modifications, or changes in the effective quark mass. Consequently, the parameters appearing in phenomenological quarkonium potentials, including the confining strength and Coulomb-like coupling terms, can acquire explicit time dependence. The present model therefore provides an effective framework for investigating dynamical quark–antiquark systems within a quantum-mechanical setting.

Based on the above description, the Hamiltonian that characterizes the motion of the pair of quarks is expressed as:

$$\begin{eqnarray}
H(t)=\frac{p^{2}}{2\mu (t)}+g(t)r^{2}-\frac{Z(t)}{r} .
\end{eqnarray}$$

This Hamiltonian enables the calculation of the quantum-mechanical spectrum including wave functions and excitation properties, which is then used to theoretically explain the mass and structure of mesons. The momentum operator in spherical coordinates is represented as $p^{2}=p_{r}^{2}+\frac{L^{2}}{r^{2}}$ , where $p_{r}=-i\hbar (\frac{\partial }{\partial r}+\frac{1}{r})$ and *L* is the total angular momentum, the formula for which is given by

$$\begin{eqnarray}
L^{2}=-\hbar ^{2}\bigg [\frac{1}{\sin ^{2}\theta }\frac{\partial ^{2}}{\partial \varphi ^{2}}+\frac{1}{\sin \theta }\frac{\partial }{\partial \theta }\bigg (\sin \theta \frac{\partial }{\partial \theta }\bigg )\bigg ].
\end{eqnarray}$$

To investigate the dynamical features of the system, we must solve the Schrödinger equation for the Hamiltonian in Eq. (1). If we denote the system’s state vector as $|\Psi (t)\rangle$ , the corresponding Schrödinger equation is of the form

$$\begin{eqnarray}
i\hbar \frac{\partial }{\partial t}|\Psi (t)\rangle =H(t)|\Psi (t)\rangle .
\end{eqnarray}$$

The time dependence of the Hamiltonian requires a specialized approach to solve this equation, as will be presented subsequently.

#### 2.1.2.

Because $H(t)$ in Eq. (1) is represented in terms of time functions, directly solving Eq. (3) can be very challenging. To overcome this difficulty, we adopt the dynamical invariant method [24]. The core idea of this method is to treat the system in terms of a nontrivial invariant operator $I(t)$ rather than $H(t)$ in Eq. (1), where $I(t)$ does not involve time-derivative operators. Moreover, $I(t)$ is closely related to the concept of invariants in classical mechanics (e.g. constants of motion), thereby facilitating interpretation from the perspective of the quantum–classical correspondence.

By the definition of the invariant operator, $I(t)$ satisfies the equation:

$$\begin{eqnarray}
\frac{dI}{dt}=\frac{\partial I}{\partial t}+\frac{1}{i\hbar }[I,H]=0.
\end{eqnarray}$$

If such an invariant exists, it is possible to put the solution of the Schrödinger equation (3) in terms of its eigenfunction $\Phi (\vec{r},t)$ , such that

$$\begin{eqnarray}
\Psi (\vec{r},t)=e^{i\alpha (t)}\Phi (\vec{r},t),
\end{eqnarray}$$

where $\alpha (t)$ is a phase function that can be obtained from

$$\begin{eqnarray}
\hbar \frac{d\alpha (t)}{dt}=\left\langle \Phi \left|\bigg (i\hbar \frac{\partial }{\partial t}-H\bigg )\right|\Phi \right\rangle .
\end{eqnarray}$$

To find the expression of the invariant $I(t)$ utilizing Eq. (4), we set

$$\begin{eqnarray}
I(t)=A(t)p^{2}+B(t)\left( rp_{r}+p_{r}r\right) +C(t)r^{2}-\frac{D(t)}{r},
\end{eqnarray}$$

where $A(t)$ , $B(t)$ , $C(t)$ , and $D(t)$ are time-dependent functions to be determined. Then, a mathematical procedure with this assumed formula of $I(t)$ after substituting Eqs. (1) and (7) into Eq. (4) yields

$$\begin{eqnarray}
A(t) =A_{0}\rho ^{2}(t),
\end{eqnarray}$$

$$\begin{eqnarray}
B(t) =-A_{0}\mu (t)\rho (t)\dot{\rho }(t),
\end{eqnarray}$$

$$\begin{eqnarray}
C(t) =A_{0}\bigg (\mu ^{2}(t)[\dot{\rho }(t)]^{2}+\frac{\Omega ^{2}}{4\rho ^{2}(t)}\bigg ),
\end{eqnarray}$$

$$\begin{eqnarray}
D(t) =2A_{0}\mu (t)Z(t)\rho ^{2}(t),
\end{eqnarray}$$

where $A_{0}$ is an arbitrary real constant and $\rho (t)$ is a solution of the following equation

$$\begin{eqnarray}
\ddot{\rho }(t)+\frac{\dot{\mu }(t)}{\mu (t) }\dot{\rho }(t)+2\frac{g(t)}{\mu (t)}\rho (t) =\frac{\Omega ^{2}}{4\mu ^{2}(t)\rho ^{3}(t)},
\end{eqnarray}$$

with $\Omega$ being a real constant. The above invariant is valid under the condition that $Z(t)$ follows the relation

$$\begin{eqnarray}
\dot{Z}(t)+\bigg (\frac{\dot{\mu }(t)}{\mu (t)}+\frac{\dot{\rho }(t)}{\rho (t)}\bigg )Z(t)=0.
\end{eqnarray}$$

Equation (7) with Eqs. (8)–(11) constitutes the complete quadratic invariant operator of the system. This operator provides a robust analytical framework for the system, in which direct solutions are difficult to obtain due to the complicated time evolution of the Hamiltonian. It also enables a deeper understanding of dynamical properties of the system, including phase transitions and topological changes, beyond merely providing a powerful methodology for obtaining solutions.

#### 2.1.3.

Because the quantum wave functions of the system are represented in terms of the eigenfunctions of $I(t)$ , it is now necessary to evaluate its eigenvalue equation. We begin by writing the eigenvalue equation of the invariant $I(t)$ as

$$\begin{eqnarray}
I(t)\Phi _{n}(\vec{r},t)=\Lambda _{n}\Phi _{n}(\vec{r},t),
\end{eqnarray}$$

where $\Lambda _{n}$ are the eigenvalues and $\Phi _{n}(\vec{r},t)$ are time-dependent eigenfunctions. Given that $I(t)$ , as defined in Eq. (7) along with Eqs. (8)–(11), has a complicated form, it is favorable to solve Eq. (14) after we transform it mathematically into a simple form. To do this, we consider a unitary transformation of the form

$$\begin{eqnarray}
\Phi _{n}^{\prime }(\vec{r})=U(t)\Phi _{n}(\vec{r},t),
\end{eqnarray}$$

where $U(t)$ is a time-dependent unitary operator given by

$$\begin{eqnarray}
U(t)=\exp \left[ \frac{i\ln \rho (t)}{2\hbar }\left( rp_{r}+p_{r}r\right) \right] \exp \left[ -i\frac{\mu (t)\dot{\rho }(t)}{2\hbar \rho (t)}r^{2}\right] .
\end{eqnarray}$$

Then, if we write the transformed invariant operator as $I_0$ , the eigenvalue equation in the transformed frame is represented in terms of $\Phi _{n}^{\prime }(\vec{r})$ , such that

$$\begin{eqnarray}
UIU^{-1}\Phi _{n}^{\prime }(\vec{r})=I_{0}\Phi _{n}^{\prime }(\vec{r})=\Lambda _{n}\Phi _{n}^{\prime }(\vec{r}).
\end{eqnarray}$$

Now, from a mathematical procedure using Eq. (16), we obtain that

$$\begin{eqnarray}
\left[ A_{0}p^{2}+A_{0}\frac{\Omega ^{2}}{4}r^{2}-D_{0}\frac{1}{r}\right] \Phi _{nlm}^{\prime }(r,\theta ,\varphi )=\Lambda _{nl}\Phi _{nlm}^{\prime }(r,\theta ,\varphi ),
\end{eqnarray}$$

where $D_{0} = 2A_{0}\mu (t)Z(t)\rho (t)$ while $n=0,1,2,\cdots$ . An infinite number of bound states may arise from the strong confinement of quarkonium, which is primarily attributed to the harmonic term dominating at large distances. By the way, the direct differentiation of $D_0$ with respect to *t* , followed by the application of Eq. (13), yields $d D_0/dt=0$ . This means that $D_0$ is in fact a constant of motion, as expected—since the operator on the left-hand side of Eq. (18) is the transformed version of the original invariant $I(t)$ . It is worth noting that Eq. (18) is very simple compared to the original eigenvalue equation represented in Eq. (14).

The existence of a perturbation of Coulomb potential may make the problem much more difficult. In what follows, the eigenvalue equation (18) can be rewritten as

$$\begin{eqnarray}
\begin{bmatrix}-\hbar ^{2}A_{0}\Big [\frac{1}{r^{2}}\frac{\partial }{\partial r}\left( r^{2}\frac{\partial }{\partial r}\right) +\frac{1}{r^{2}\sin ^{2}\theta }\frac{\partial ^{2}}{\partial \varphi ^{2}} \\+\frac{1}{r^{2}\sin \theta }\frac{\partial }{\partial \theta }(\sin \theta \frac{\partial }{\partial \theta })\Big ]+A_{0}\frac{\Omega ^{2}}{4}r^{2}-\frac{D_{0}}{r} \end{bmatrix} \Phi _{nlm}^{\prime }(r,\theta ,\varphi )=\Lambda _{nl}\Phi _{nlm}^{\prime }(r,\theta ,\varphi ).
\end{eqnarray}$$

According to the invariant operator theory, the wave functions of the transformed system are represented in terms of $\Phi _{nlm}^{\prime }(r,\theta ,\varphi )$ . Because this equation is independent of time, we can apply the method of separation of variables in order to solve it. Considering this, we take

$$\begin{eqnarray}
\Phi _{nlm}^{\prime }(r,\theta ,\varphi )=R_{nl}(r)Y_{lm}(\theta ,\varphi )=\frac{u_{nl}(r)Y_{lm}(\theta ,\varphi )}{r},
\end{eqnarray}$$

where we have introduced $u_{nl}(r) = r R_{nl}(r)$ , whereas $Y_{lm}(\theta ,\varphi )$ are the spherical harmonics. The angular part of the eigenfunctions takes the form

$$\begin{eqnarray}
Y_{lm}(\theta ,\varphi )= N_{lm}e^{im\varphi }P_{l}^{m}(\cos \theta ),
\end{eqnarray}$$

where $P_{l}^{m}(x)$ are the associated Legendre polynomials, while the spherical normalization constants are $N_{lm}=(-1)^m \lbrace [(2l+1)/(4\pi )](l-m)!/(l+m)! \rbrace ^{1/2}$ . The allowed orbital and magnetic quantum numbers are $l=0,1,2,\cdots$ and $m=-l,-l+1,\cdots ,l$ , respectively.

Some algebraic manipulation, after substituting Eq. (20) into Eq. (19), leads to the following radial equation:

$$\begin{eqnarray}
\frac{d^{2}u_{nl}(r)}{dr^{2}}+\left( \frac{\Lambda _{nl}}{\hbar ^{2}A_{0}}-\frac{\Omega ^{2}}{4\hbar ^{2}}r^{2}+\frac{D_{0}}{\hbar ^{2}A_{0}}\frac{1}{r}-\frac{l(l+1)}{r^{2}}\right) u_{nl}(r)=0.
\end{eqnarray}$$

The equation is analytically intractable unless $\Omega = 0$ owing to the coexistence of $r^{2}$ and $-\frac{1}{r^{2}}$ terms. To resolve this difficulty, we employ the Greene–Aldrich approximation scheme of the form [36–39]

$$\begin{eqnarray}
\frac{1}{r^{2}} \simeq \frac{\delta ^{2}}{\left( 1-e^{-\delta r}\right) ^{2}},~~~~~\frac{1}{r} \simeq \frac{\delta }{1-e^{-\delta r}},
\end{eqnarray}$$

together with a coordinate transformation $y=e^{-\delta r}$ . This type of approximation is commonly used to describe particles governed by a screened potential, where the parameter $\delta$ is typically chosen to be equal to the screening mass [29,36]. Though the potential in our system is not screened, choosing $\delta$ to be similar to the typical screening mass still provides a good approximation, particularly accurate for distances comparable to or smaller than $1/\delta$ . For example, in quarkonium models, $\delta$ is usually taken to be in the range of approximately 200 to 900 MeV [40,41]. Thus, selecting $\delta$ within this range is a reasonable choice.

Then, by rewriting the notation as $u_{nl}(r)\rightarrow U_{nl}(y)$ , the differential equation (22) becomes

$$\begin{eqnarray}
\frac{d^{2}U_{nl}(y)}{dy^{2}}+\frac{\left( 1-y\right) }{y(1-y)}\frac{dU_{nl}(y)}{dy}+\frac{1}{y^{2}(1-y)^{2}}\begin{bmatrix}-\left( \varepsilon _{nl}-6P\right) y^{2} \\+\left( 2\varepsilon _{nl}-4P+Q\right) y \\-\left( \varepsilon _{nl}-P+Q+\gamma \right) \end{bmatrix} U_{nl}(y)=0,
\end{eqnarray}$$

where

$$\begin{eqnarray}
\varepsilon _{nl}=\frac{-\Lambda _{nl}}{\hbar ^{2}A_{0}\delta ^{2}},~~~P=-\frac{\Omega ^{2}}{4\hbar ^{2}\delta ^{4}},~~~Q=-\frac{D_{0}}{\hbar ^{2}A_{0}\delta },~~~\gamma =l(l+1).
\end{eqnarray}$$

In the representation of Eq. (24), we neglected higher-order terms corresponding to $y^{3}$ and $y^{4}$ . To solve this equation, we use the NUFA method [29] outlined in the next subsection.

### 2.2.

A second-order differential equation of hypergeometric type can be solved in a simple and elegant way by using the so-called NUFA method. This is an improvement over the parametric NU method, which is relatively cumbersome due to the need to find the square of the polynomials and other conditions required. With the NUFA method, many diffierential equations, including our wave equation, can be properly tackled, allowing for the identification of singularities and the determination of the eigenvalue spectrum. As a general type of second-order differential equation, which encompasses Eq. (24), we consider the following equation [31]:

$$\begin{eqnarray}
\left[ \frac{d^{2}}{ds^{2}}+\frac{\alpha _{1}-\alpha _{2}s}{s(1-\alpha _{3}s)}\frac{d}{ds}+\frac{-\zeta _{1}s^{2}+\zeta _{2}s-\zeta _{3}}{s^{2}(1-\alpha _{3}s)^{2}}\right] \phi (s)=0,
\end{eqnarray}$$

where $\alpha _{i}$ $(i=1,2,3)$ and $\zeta _{i}$ are parameters appropriately chosen depending on a given system. Because this equation has two singularities at $s\rightarrow 0$ and $s\rightarrow 1/\alpha _{3}$ , it is possible to take the solution in the form

$$\begin{eqnarray}
\phi (s)=s^{\lambda }(1-\alpha _{3}s)^{\upsilon }f(s).
\end{eqnarray}$$

By substituting this formula into Eq. (26) along with the following choice,

$$\begin{eqnarray}
\lambda =\frac{1}{2}\lbrace 1-\alpha _{1}\pm [\left( 1-\alpha _{1}\right) ^{2}+4\zeta _{3}]^{1/2}\rbrace ,
\end{eqnarray}$$

$$\begin{eqnarray}
\upsilon &=&\frac{1}{2\alpha _{3}} \lbrace \alpha _{3}+\alpha _{1}\alpha _{3}-\alpha _{2}\pm [\left( \alpha _{3}+\alpha _{1}\alpha _{3}-\alpha _{2}\right) ^{2} \\&&+4\alpha _3 (\zeta _{3}\alpha _{3}-\zeta _{2}+\zeta _{1}/\alpha _{3})]^{1/2}\rbrace ,
\end{eqnarray}$$

we have [29]

$$\begin{eqnarray}
s(1-\alpha _{3}s)\frac{d^{2}f(s)}{ds^{2}}+\kappa \frac{df(s)}{ds} -\alpha _{3} \kappa _+ \kappa _- f(s)=0,
\end{eqnarray}$$

where

$$\begin{eqnarray}
\kappa = \alpha _{1}+2\lambda -(2\lambda \alpha _{3}+2\upsilon \alpha _{3}+\alpha _{2})s,
\end{eqnarray}$$

$$\begin{eqnarray}
\kappa _\pm = \lambda +\upsilon +\frac{1}{2}\left( \frac{\alpha _{2}}{\alpha _{3}}-1\right) \pm \bigg [ \frac{1}{4}\left( \frac{\alpha _{2}}{\alpha _{3}}-1\right) ^{2}+\frac{\zeta _{1}}{\alpha _{3}^{2}} \bigg ]^{1/2}.
\end{eqnarray}$$

Thus, the equation is reduced to a tractable form.

We now examine a simple case where $\alpha _3=1$ , which meets the equation associated with our considered system. Under the condition that the principal quantum number takes the form

$$\begin{eqnarray}
n=- \kappa _- ,
\end{eqnarray}$$

the function $f(s)$ is represented by the Gauss hypergeometric function as [29]

$$\begin{eqnarray}
f (s) = {_{2}F_{1}}(a,b;c;s),
\end{eqnarray}$$

where $a=\kappa _-$ , $b=\kappa _+$ , and $c=\alpha _{1}+2\lambda$ . The mathematical formula of this hypergeometric function is as follows: $_{2}F_{1}(a,b;c;s)=\sum _{k=0}^{\infty }\frac{\left( a\right) _{k}\left( b\right) _{k}}{\left( c\right) _{k}k!}s^{k}$ , where $\left( a\right) _{k}=a(a+1)(a+2)\cdots (a+k-1)$ for $k\ge 1$ and $\left( a\right) _{0}=1$ .

The Gauss hypergeometric function $_{2}F_{1}(a,b;c;s)$ serves as a fundamental and unifying special function that encompasses many other functions as special or limiting cases [42–45]. As defined in Eq. (34) with Eq. (33), the function $_{2}F_{1}(a,b;c;s)$ terminates and reduces to a polynomial when *a* is zero or a negative integer, a case that is often associated with Laguerre polynomials. In particular, $_{2}F_{1}(a,b;c;s)$ is a generalization of Kummer’s confluent hypergeometric function $_{1}F_{1}(a;c;s)$ . Hence, in the limiting case where $b \rightarrow \infty$ , $_{2}F_{1}(a,b;c;s)$ converges to $_{1}F_{1}(a;c;s)$ . This convergence is rooted in the fact that, while ${}_2F_1$ has two regular singular points, sending one of them to infinity (a process known as confluence) effectively merges the singularities, giving rise to ${}_1F_1$ [44]. Additionally, the Jacobi polynomials can also be expressed in terms of the Gauss hypergeometric function (for a typical representation, see formula 15.9.1 of Ref. [45]). The ${}_2F_1$ function thus plays a major role in constructing analytical solutions and enables rigorous, systematic analysis across a broad spectrum of scientific disciplines beyond mathematics and physics. The method presented in this subsection, which naturally leads to the appearance of the ${}_2F_1$ function as a solution, is applicable to the analysis of a wide class of physical systems governed by one or more interaction potentials. It is, of course, well suited to handling Eq. (24).

### 2.3.

#### 2.3.1.

We now proceed to derive the wave functions of the system. For clarity and better reader understanding, we briefly summarize our strategy at this point. Because, in our time-dependent case, the wave functions are described by the eigenfunctions of $I(t)$ (instead of $H(t)$ ), we need to determine the eigenfunctions $\Phi _n$ of Eq. (14). However, directly solving for $\Phi _n$ is likely to be a formidable task. To circumvent this difficulty, we first derive $\Phi _n^{\prime }$ from Eq. (17), and then reconstruct the desired eigenfunctions $\Phi _n$ by inversely transforming the obtained $\Phi _n^{\prime }$ , utilizing the inverse relation of Eq. (15) (for the inverse relation, see Eq. (46), which appears later). Once $\Phi _n$ are obtained, the wave functions $\Psi _n$ can be expressed in terms of them accordingly.

To obtain $\Phi _n^{\prime }$ , we need to solve Eq. (24) by setting the associated functions, in accordance with Eq. (27), as

$$\begin{eqnarray}
U_{nl}(y)=N_{nl}y^{\lambda }(1-y)^{\upsilon }f_{nl}(y),
\end{eqnarray}$$

where $N_{nl}$ are normalization constants for the final wave functions. We applied $\alpha _{3}=1$ in this equation. In addition, from the comparison of Eq. (24) with the NUFA equation (26), we acquire the following parameter relations:

$$\begin{eqnarray}
&&\alpha _{1}=\alpha _{2}=1,\quad\quad\quad\zeta _{1}=\varepsilon _{nl}-6P,~~~~~~ \\&&\zeta _{2}=2\varepsilon _{nl}-4P+Q,\quad\quad\zeta _{3}=\varepsilon _{nl}-P+Q+\gamma .~~~~~~
\end{eqnarray}$$

Furthermore, the singularity exponents reduce to

$$\begin{eqnarray}
\lambda =\sqrt{\varepsilon _{nl}-P+Q+\gamma },\quad\quad\upsilon =\frac{1}{2}+\sqrt{\left( l+\frac{1}{2}\right) ^{2}-3P}.~~~~~~
\end{eqnarray}$$

Although the original NUFA representations of $\lambda$ and $\upsilon$ , given in Eqs. (28) and (29), include both positive and negative signs ( $\pm$ ) from a purely mathematical perspective, we choose to retain only the positive sign in this case as can be seen from Eq. (37). This selection is due to the fundamental physical constraint, which is that the wave functions must remain finite throughout the entire range of *r* and must vanish as $r \rightarrow 0$ and $r \rightarrow \infty$ .

The expansion of Eq. (33) under the first condition in Eq. (36) yields

$$\begin{eqnarray}
n+\lambda +\upsilon = \sqrt{\zeta _1}.
\end{eqnarray}$$

By squaring both sides of this equation and using the second relation from Eq. (36) together with the formula of $\lambda$ given in Eq. (37), the equation associated with the eigenvalues can be obtained as

$$\begin{eqnarray}
2[\varepsilon _{nl}-P+Q+\gamma ]^{1/2}(n+\upsilon )+(n+\upsilon )^{2}+5P+Q+\gamma =0.
\end{eqnarray}$$

It is possible to rearrange this equation into a formula for $\varepsilon _{nl}$ by first isolating the square root term on the left-hand side, shifting all other terms to the right-hand side, and then squaring both sides. Substituting the expression for $\upsilon$ from Eq. (37) into the resulting equation, and using Eq. (25), we finally obtain the eigenvalues:

$$\begin{eqnarray}
\Lambda _{nl} &=& l(l+1)\hbar ^{2}A_{0}\delta ^{2}+\frac{A_{0}\Omega ^{2}}{4\delta ^{2}}-D_{0}\delta \\&&-\frac{\hbar ^{2}A_{0}\delta ^{2}}{4}\left[ \frac{l(l+1)-\frac{5\Omega ^{2}}{4\hbar ^{2}\delta ^{4}}-\frac{D_{0}}{\hbar ^{2}A_{0}\delta }+\left( n+\upsilon \right) ^{2}}{n+\upsilon }\right] ^{2}.
\end{eqnarray}$$

The eigenvalues obtained in Eq. (40) depend explicitly on several parameters of the system, including the effective oscillator frequency and coupling constants, which may exhibit time dependence through the Hamiltonian. This feature reflects the fact that the present model belongs to the class of parameter-dependent quantum systems. In such systems, variations of the energy spectrum with respect to the relevant parameters are conceptually related to the Hellmann–Feynman theorem [46,47], which expresses derivatives of eigenvalues in terms of expectation values of derivatives of the Hamiltonian. Although the present work does not explicitly employ the Hellmann–Feynman theorem in deriving Eq. (40), the obtained spectrum is consistent with the general framework of parameter-dependent quantum mechanics.

In addition, by considering Eq. (27) with Eqs. (28), (29), and (34), the corresponding radial wave functions can be derived in terms of $\Lambda _{nl}$ to be

$$\begin{eqnarray}
U_{nl}(y) &=& N_{nl}y^{\sqrt{-\frac{\Lambda _{nl}}{\hbar ^{2}A_{0}\delta ^{2}}+\frac{\Omega ^{2}}{4\hbar ^{2}\delta ^{4}}-\frac{D_{0}}{\hbar ^{2}A_{0}\delta }+l(l+1)}} \\&&\times {(1-y)^{\frac{1}{2}+\sqrt{\left( l+\frac{1}{2}\right) ^{2}+\frac{3\Omega ^{2}}{4\hbar ^{2}\delta ^{4}}}}}{_{2}F_{1}}(\bar{a},\bar{b};\bar{c};y),
\end{eqnarray}$$

where the parameters are defined as

$$\begin{eqnarray}
\bar{a}=\lambda +\upsilon -\sqrt{\varepsilon _{nl}-6P},
\end{eqnarray}$$

$$\begin{eqnarray}
\bar{b}=\lambda +\upsilon +\sqrt{\varepsilon _{nl}-6P},
\end{eqnarray}$$

$$\begin{eqnarray}
\bar{c}=1+2\lambda .
\end{eqnarray}$$

From a mathematical standpoint, it is possible to define the parameter $\bar{a}$ interchangeably with $\bar{b}$ . However, by choosing $\bar{a}$ as in Eq. (42), we allow the possibility for $\bar{a}$ to take on negative values. When $\bar{a}$ is a nonpositive integer (i.e. a negative integer or zero), the Gauss hypergeometric function in Eq. (41) terminates at the ( $n+1$ )th term and reduces to a polynomial. Through the use of Eq. (20) with Eq. (41), the full eigenfunctions of the invariant operator $I_{0}$ are represented in the form

$$\begin{eqnarray}
\Phi _{nlm}^{\prime }(r,\theta ,\varphi ) &=& N_{nl}\frac{1}{r}e^{-\left( \sqrt{-\frac{\Lambda _{nl}}{\hbar ^{2}A_{0}\delta ^{2}}+\frac{\Omega ^{2}}{4\hbar ^{2}\delta ^{4}}-\frac{D_{0}}{\hbar ^{2}A_{0}\delta }+l(l+1)}\right) \delta r} \\&&\times \left( 1-e^{-\delta r}\right) ^{\frac{1}{2}+\sqrt{\left( l+\frac{1}{2}\right) ^{2}+\frac{3\Omega ^{2}}{4\hbar ^{2}\delta ^{4}}}}{_{2}F_{1}}(\bar{a},\bar{b};\bar{c};e^{-\delta r})Y_{lm}(\theta ,\varphi ).
\end{eqnarray}$$

The complete eigenfunctions of the invariant *I* that belongs to the original system, i.e., the untransformed system, are given by the relation

$$\begin{eqnarray}
\Phi _{nlm}(\vec{r},t)=U^{-1}(t)\Phi _{nlm}^{\prime }(\vec{r}).
\end{eqnarray}$$

The evaluation of this using Eqs. (16) and (45) results in

$$\begin{eqnarray}
\Phi _{nlm}(r,\theta ,\varphi ,t) &=& N_{nl} \frac{1 }{\sqrt{\rho (t)}r} \exp \left( \frac{i\mu (t)\dot{\rho }(t)}{2\hbar \rho (t)}r^{2}\right) \\&&\times e^{-\left( \sqrt{-\frac{\Lambda _{nl}}{\hbar ^{2}A_{0}\delta ^{2}}+\frac{\Omega ^{2}}{4\hbar ^{2}\delta ^{4}}-\frac{D_{0}}{\hbar ^{2}A_{0}\delta }+l(l+1)}\right) \delta \frac{r}{\rho (t) }} \\&&\times \left( 1-e^{-\delta \frac{r}{\rho (t) }}\right) ^{\frac{1}{2}+\sqrt{\left( l+\frac{1}{2}\right) ^{2}+\frac{3\Omega ^{2}}{4\hbar ^{2}\delta ^{4}}}}{_{2}F_{1}}(\bar{a},\bar{b};\bar{c};e^{-\delta \frac{r}{\rho (t) }})Y_{lm}(\theta ,\varphi ).
\end{eqnarray}$$

Meanwhile, $N_{nl}$ are derived by normalizing these resultant eigenfunctions. The normalization procedure yields (see Appendix A for a detailed evaluation)

$$\begin{eqnarray}
N_{nl}=\sqrt{\delta } \frac{\Gamma (2(\upsilon +\lambda )+n)}{n!\Gamma (2\lambda +1)} \Bigg [ \sum _{k=0}^n\sum _{k^{\prime }=0}^n (-1)^{k+k^{\prime }} \mathcal {N}_{nl}(k,k^{\prime }) \Bigg ]^{-1/2},
\end{eqnarray}$$

where

$$\begin{eqnarray}
\mathcal {N}_{nl}(k,k^{\prime }) &=& \frac{\Gamma (2(\upsilon +\lambda )+n+k) \Gamma (2(\upsilon +\lambda )+n+k^{\prime })}{k!k^{\prime }!(n-k)!(n-k^{\prime })!\Gamma (2\lambda +k+1)\Gamma (2\lambda +k^{\prime }+1)} \\&& \times {\mathrm{B}}(k+k^{\prime }+2\lambda ,2\upsilon +1),
\end{eqnarray}$$

whereas ${\mathrm{B}}(a_1,a_2)$ denotes the Beta function [48]. The eigenfunctions in Eq. (47) are essentially related to the Schrödinger solutions of the system. We demonstrate how to construct the full quantum solutions in terms of these eigenfunctions in the following subsection.

#### 2.3.2.

According to Lewis–Riesenfeld theory [24], the wave functions of our time-dependent system are represented in terms of the eigenfunctions $\Phi _{nlm}(\vec{r},t)$ of the invariant operator, which are given in Eq. (47). However, these eigenfunctions alone do not fully define the time-dependent wave functions. Another required factor is the phase $\alpha (t)$ associated with each quantum state. We begin from Eq. (6) to derive this factor. Based on the unitary relation given in Eq. (46), Eq. (6) can be expressed in terms of the transformed eigenstate $\vert \Phi ^\prime \rangle$ instead of $\vert \Phi \rangle$ , such that

$$\begin{eqnarray}
\hbar \frac{d\alpha (t)}{dt}=\left\langle \Phi ^{\prime }\left|\Bigg ( i \hbar \frac{\partial }{\partial t}-\frac{I_{0}}{2A_{0}\mu (t)\rho ^{2}(t)} \Bigg ) \right|\Phi ^{\prime }\right\rangle .
\end{eqnarray}$$

Then, a minor evaluation using Eq. (18) yields the phases in the form

$$\begin{eqnarray}
\alpha _{nl}(t)=-\frac{\Lambda _{nl}}{2A_{0}\hbar }\int _{0}^{t}\frac{1}{\mu (t^{\prime })\rho ^{2}(t^{\prime })}dt^{\prime }+\alpha _{nl}(0),
\end{eqnarray}$$

where $\Lambda _{nl}$ are given by Eq. (40). These phases emerge as a result of the time evolution of the wave functions and can produce observable consequences, such as interference patterns, through relative phase differences between quantum states. Both the geometrical phase and the conventional dynamical phase contribute to the total phase. While the dynamical phase results from the cumulation of energy over time, the geometric phase originates from the intrinsic geometry of the path traced by the state vector during evolution, often under adiabatic conditions. The significance of the geometrical phase lies in the fact that it is not a mathematical artifact but a physically meaningful, gauge-invariant quantity that can influence the actual quantum state of the system. It not only takes place in this system, but appears ubiquitously across various physical systems as well, and can be measured via interference experiments. Unlike the dynamical phase, which depends on the eigenvalues of the Hamiltonian, the geometrical phase depends on the trajectory of the eigenvectors themselves. As such, it plays a critical role in associated phenomena such as energy level splitting and the orbital behavior of quasiparticles, making it a subject of continued theoretical and experimental interest. In particular, in systems with irregular and nonlinear interactions, such as quark–antiquark systems, it reflects the underlying symmetries, binding mechanisms, and topological features of the system [49,50]. Accordingly, the geometrical phase constitutes a fundamental component in the evolution of the system, linking theoretical descriptions to experimental observations.

The solutions of the original Schrödinger equation (3) corresponding to the Hamiltonian in Eq. (1) can now be written in configuration space as

$$\begin{eqnarray}
\Psi _{nlm}(r,\theta ,\varphi ,t)=e^{i\alpha _{nl}(t)}\Phi _{nlm}(r,\theta ,\varphi ,t),
\end{eqnarray}$$

where the explicit forms of $\alpha _{nl}(t)$ and $\Phi _{nlm}(r,\theta ,\varphi ,t)$ are given in Eqs. (51) and (47), respectively. The resulting wave functions (52), together with $\Lambda _{nl}$ in Eq. (40), serve as fundamental solutions for the quark–antiquark systems under consideration. The obtained wave functions enable us to compute spatial distributions (densities), transition probabilities, correlations, etc. Consequently, they provide a foundational framework for analyzing these systems, advancing our perspective on the interaction mechanisms between quarks and antiquarks, particularly in the context of quantum effects. The spatial structure of the wave functions enables estimation of key physical properties, such as the size of the particle (e.g. root-mean-square radius) and the binding strength of the bound state, which are critical for characterizing the complexity of strong interaction systems [51,52]. These wave functions thus serve as important tools for bridging theoretical predictions and experimental results, helping reveal the internal structure, confinement dynamics, and symmetry properties of the quark–antiquark pair.

## 3.

In this study, we investigated the interaction between a quark and an antiquark by modeling its mechanical behavior using a nonstationary 3D harmonic oscillator coupled with a Coulomb potential. To solve the associated Schrödinger equation, we employed the recently refined NUFA method, together with the invariant operator method and the unitary transformation approach. The NUFA method, based on a novel conceptual framework, allowed us to bypass the complex mathematical manipulations commonly encountered in other techniques. As a result, we obtained the wave functions of the system in closed form, expressed in terms of the Gauss hypergeometric function. These solutions offer advantages over previous ones that employed biconfluent Heun functions, as Gauss hypergeometric functions are among the most thoroughly characterized and best understood special functions in mathematics.

Our solutions are essential for analyzing quark–antiquark pairs, which form the core of mesons and thus play a central role in the strong interaction. Although their dynamics are governed by QCD, the theory exhibits intricate behavior that significantly deviates from classical, nonrelativistic mechanics, encompassing numerous unresolved challenges. These include the incomplete theoretical understanding of quark confinement and hadronization mechanisms, the precise determination of quark masses and meson spectra, and the possible existence of exotic hadrons featuring internal structure beyond conventional quark–antiquark configurations [53,54]. Additional open questions involve the quantitative description of the quark condensate mechanism, the strong CP problem and the role of the $\theta$ -term, as well as the difficulty in accurately predicting the spectral properties and decay modes of heavy mesons such as $J/\psi$ [5,55]. The analytical solutions developed in this study may provide insights toward addressing these longstanding issues.

## Funding

Open Access funding: SCOAP 3 .

Although normalization is formally defined for the full wave functions, the phase factors do not affect the outcome. Therefore, in the normalization process, it suffices to consider only Eq. (47), which corresponds to the eigenfunctions. The normalization condition associated with Eq. (47) is given by

$$\begin{eqnarray}
\int _0^{2\pi } \int _0^\pi \int _0^\infty \left|\Phi _{nlm} (r,\theta ,\varphi ,t) \right|^{2}r^{2} \sin \theta dr d\theta d\varphi =1.
\end{eqnarray}$$

Because the angular part $Y_{lm} (\theta ,\varphi )$ , defined in Eq. (21), is already normalized, we only need to consider the radial part in this evaluation. Thus, based on the radial part of Eq. (47), Eq. (A.1) can be rewritten as

$$\begin{eqnarray}
\frac{\left|N_{nl}\right|^{2}}{\rho (t) }\int _{0}^{\infty }e^{-2\lambda \delta \frac{r}{\rho (t)}}\left( 1-e^{-\delta \frac{r}{\rho (t)}}\right) ^{2\upsilon }\left[ {_{2}F_{1}}(\bar{a},\bar{b};\bar{c};e^{-\delta \frac{r}{\rho (t)}})\right] ^{2}dr =1.
\end{eqnarray}$$

To simplify the integral, we introduce the substitution

$$\begin{eqnarray}
z=e^{-\delta \frac{r}{\rho (t)}},
\end{eqnarray}$$

and consider the fact that $\bar{a}$ and $\bar{b}$ in this case can be expressed in terms of *n* as

$$\begin{eqnarray}
\bar{a}=-n,~~~~~~~ \bar{b}=2(\upsilon +\lambda )+n,
\end{eqnarray}$$

along with Eq. (44) as the formula of $\bar{c}$ . Then, Eq. (A.2) becomes

$$\begin{eqnarray}
\frac{\left|N_{nl}\right|^{2}}{\delta }g_{\rm int}=1,
\end{eqnarray}$$

where $g_{\rm int}$ is the integral part:

$$\begin{eqnarray}
g_{\rm int}=\int _{0}^{1}z^{2\lambda -1} \left( 1-z\right) ^{2\upsilon }\left[ {_{2}F_{1}}(-n,2(\upsilon +\lambda )+n;2\lambda +1;z)\right] ^{2}dz .
\end{eqnarray}$$

To evaluate the integration $g_{\rm int}$ , we use the expansion formula for the hypergeometric function (see formula 15.2.4 of Ref. [45]):

$$\begin{eqnarray}
{_{2}F_{1}}(-n,b;c;s)=\sum _{k=0}^n (-1)^k \binom{n}{k} \frac{(b)_k}{(c)_k}s^k,
\end{eqnarray}$$

which holds when $n=0,1,2,\cdots$ and $c \ne 0,-1,-2,\cdots$ . This relation also holds when $c=-n-j$ , where $j=0,1,2,\cdots$ . Substituting this series expansion into Eq. (A.6), we obtain

$$\begin{eqnarray}
g_{\rm int}&=&\sum _{k=0}^n\sum _{k^{\prime }=0}^n(-1)^{k+k^{\prime }}\binom{n}{k} \binom{n}{k^{\prime }} \frac{(2(\upsilon +\lambda )+n)_k (2(\upsilon +\lambda )+n)_{k^{\prime }}}{(2\lambda +1)_k(2\lambda +1)_{k^{\prime }}} \\&&\times \int _{0}^{1}z^{k+k^{\prime }+2\lambda -1} \left( 1-z\right) ^{2\upsilon }dz .
\end{eqnarray}$$

The integral in the above equation is the Beta function, which is defined as [48]

$$\begin{eqnarray}
\mathrm{B}(a_1,a_2)= \int _0^1 s^{a_1-1} (1-s)^{a_2-1} ds,
\end{eqnarray}$$

under the conditions ${\rm Re}a_1\gt 0$ and ${\rm Re}a_2\gt 0$ . The two shape parameters of this function, in our case, are $a_1=k+k^{\prime }+2\lambda$ and $a_2=2\upsilon +1$ . Thus, considering that the Pochhammer number is given by $(a)_k=\Gamma (a+k)/\Gamma (a)$ , where $a \ne 0,-1,-2,\cdots$ , a rearrangement of Eq. (A.5) with Eq. (A.8) leads directly to the expression for the normalization constants $N_{nl}$ , as presented in Eq. (48) of the text.

## References

- H.-T. Ding, O. Kaczmarek, A.-L. Kruse, R. Larsen, L. Mazur, S. Mukherjee, H. Ohno, H. Sandmeyer, H.-T. Shu, Nucl. Phys. A. 982, 715 (2019).10.1016/j.nuclphysa.2018.09.075
- É. Chapon et al., Prog. Part. Nucl. Phys. 122, 103906 (2022).10.1016/j.ppnp.2021.103906
- H. Bahl, S. Koren, L.-T. Wang, Eur. Phys. J. C. 84, 1100 (2024).10.1140/epjc/s10052-024-13411-3
- M. Doser, G. Farrar, G. Kornakov, Eur. Phys. J. C. 83, 1149 (2023).10.1140/epjc/s10052-023-12319-8
- T. Li, X.-D. Ma, M. A. Schmidt, R.-J. Zhang, Phys. Rev. D. 104, 035024 (2021).10.1103/PhysRevD.104.035024
- E. Oks, New Astron. Rev. 93, 101632 (2021).10.1016/j.newar.2021.101632
- Q. Bonnefoy, L. Hall, C. A. Manzari, A. McCune, C. Scherb, Phys. Rev. D. 109, 055045 (2024).10.1103/PhysRevD.109.055045
- S. Catto, F. Gürsey, Nuovo Cimento A. 99, 685 (1988).10.1007/BF02730633
- S. Catto, H. Y. Cheung, F. Gürsey, Mod. Phys. Lett. A. 6, 3485 (1991).10.1142/S0217732391004024
- A. S. de Castro, Phys. Lett. A. 346, 71 (2005).10.1016/j.physleta.2005.07.065
- Y.-S. Choun, S.-J. Sin, Prog. Theor. Exp. Phys. 2021, 013A01 (2021).10.1093/ptep/ptaa157
- H. S. Vieira, V. B. Bezerra, J. Math. Phys. 56, 092501 (2015).10.1063/1.4930871
- D. M. Reyna-Muñoz, M. A. Reyes, A. F. Téllez, Rev. Mex. Fis. E. 21, 010209 (2024).
- A. Ishkhanyan, K.-A. Suominen, J. Phys. A: Math. Gen. 34, 6301 (2001).10.1088/0305-4470/34/32/309
- F. Caruso, J. Martins, V. Oguri, Ann. Phys. 347, 130 (2014).10.1016/j.aop.2014.04.023
- J. Karwowski, H. A. Witek, Theor. Chem. Acc. 133, 1494 (2014).10.1007/s00214-014-1494-5
- A. Roseau, Bull. Belg. Math. Soc. 9, 321 (2002).
- E. R. Arriola, A. Zarzo, J. S. Dehesa, J. Comput. Appl. Math. 37, 161 (1991).10.1016/0377-0427(91)90114-Y
- E. S. Cheb-Terrab, J. Phys. A: Math. Gen. 37, 9923 (2004).10.1088/0305-4470/37/42/007
- P. Bagchi, A. Das, A. P. Mishra, A. K. Panda, Mod. Phys. Lett. A. 41, 2650051 (2026).10.1142/S0217732326500513
- Y. Akamatsu, Prog. Part. Nucl. Phys. 123, 103932 (2022).10.1016/j.ppnp.2021.103932
- T. Hayata, K. Nawa, T. Hatsuda, Phys. Rev. D. 87, 101901(R) (2013).10.1103/PhysRevD.87.101901
- H. R. Lewis Jr, Phys. Rev. Lett. 18, 510 (1967).10.1103/PhysRevLett.18.510
- H. R. Lewis Jr, W. B. Riesenfeld, J. Math. Phys. 10, 1458 (1969).10.1063/1.1664991
- S. Menouar, J. R. Choi, AIP Adv. 6, 095110 (2016).10.1063/1.4962995
- J. R. Choi, Chin. Phys. C. 35, 233 (2011).10.1088/1674-1137/35/3/005
- J.-H. Mun, H. Sakai, D.-E. Kim, Int. J. Mol. Sci. 22, 8514 (2021).10.3390/ijms22168514
- S. Menouar, J. R. Choi, Ann. Phys. 353, 307 (2015).10.1016/j.aop.2014.11.014
- A. N. Ikot, U. S. Okorie, P. O. Amadi, C. O. Edet, G. J. Rampho, R. Sever, Few-Body Syst. 62, 1 (2021).
- A. F. Nikiforov, V. B. Uvarov, Special Functions of Mathematical Physics (Birkhäuser, Boston, MA, 1988), p. 205.10.1007/978-1-4757-1595-8
- C. Tezcan, R. Sever, Int. J. Theor. Phys. 48, 337 (2009).10.1007/s10773-008-9806-y
- S. Medjber, H. Bekkar, S. Menouar, J. R. Choi, Adv. Math. Phys. 2016, 3693572 (2016).10.1155/2016/3693572
- J. R. Choi, S. Menouar, S. Medjber, H. Bekkar, J. Phys. Commun. 1, 052001 (2017).10.1088/2399-6528/aa83f6
- S. Menouar, J. R. Choi, R. Sever, Nonlinear Dyn. 92, 659 (2018).10.1007/s11071-018-4081-9
- M. Abramowitz, I. A. Stegun, Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables (Dover, New York, 1972).
- R. L. Greene, C. Aldrich, Phys. Rev. A. 14, 2363 (1976).10.1103/PhysRevA.14.2363
- R. Horchani, E. Omugbe, I. J. Njoku, L. M. Pérez, C. A. Onate, A. Jahanshir, E. Feddi, K. O. Emeje, E. S. Eyube, Sci. Rep. 14, 28582 (2024).10.1038/s41598-024-80123-9
- C. Ertugay, C. O. Edet, A. N. Ikot, B. C. Lütfüoğlu, Mol. Phys. 123, e2411327 (2025).10.1080/00268976.2024.2411327
- A. N. Ikot, U. S. Okorie, R. Sever, G. J. Rampho, Eur. Phys. J. Plus, 134, 386 (2019).
- D.-H. Lin, F.-X. Xie, Acta Phys. Sin. 33, 1569 (1984).10.7498/aps.33.1569
- B.-Q. Li, K.-T. Chao, Commun. Theor. Phys. 52, 653 (2009).10.1088/0253-6102/52/4/20
- J. Dereziński, Ann. Henri Poincaré. 15, 1569 (2014).10.1007/s00023-013-0282-4
- W. Koepf, Hypergeometric Summation–An Algorithmic Approach to Summation and Special Function Identities (Springer, London, 2014).10.1007/978-1-4471-6464-7
- G. Kristensson, Confluent hypergeometric functions, in Second Order Differential Equations: Special Functions and Their Classification (Springer, New York, 2010),Chap. 7, p. 123.10.1007/978-1-4419-7020-6\_7
- F. W. J. Olver, D. W. Lozier, R. F. Boisvert, C. W. Clark, NIST Handbook of Mathematical Functions (Cambridge University Press, New York, 2010), p. 385.(formula 15.2.4) and p. 393 (formula 15.9.1).
- H. Hellmann, Einführung in die Quantenchemie (F. Deuticke, Leipzig, 1937).
- R. P. Feynman, Phys. Rev. 56, 340 (1939).10.1103/PhysRev.56.340
- I. S. Gradshteyn, I. M. Ryzhik, Table of Integrals, Series, and Products (Elsevier, Amsterdam, 2007), 7th ed., p. 908 (formula 3.380(1)) or p. 315 (formula 3.191(3)).
- S. Banerjee, M. Dorband, J. Erdmenger, A.-L. Weigel, J. High Energy Phys. 2023, 026 (2023).10.1007/JHEP10(2023)026
- T. Asselmeyer-Maluga, M. Lulli, A. Marcianò, R. Pasechnik, E. Zappala, arXiv:2408.15986v3 [hep-th] [Search inSPIRE].
- M. Alberg, G. A. Miller, Phys. Rev. C. 110, L042201 (2024).10.1103/PhysRevC.110.L042201
- G. Ganbold, EPJ Web Conf. 3, 03014 (2010).10.1051/epjconf/20100303014
- G. F. de Téramond, S. J. Brodsky, Int. J. Mod. Phys. A. 39, 2441007 (2024).
- N. Hüsken, E. S. Norella, I. Polyakov, Mod. Phys. Lett. A. 40, 2530002 (2025).
- ALICE Collaboration, Eur. Phys. J. C. 84, 813 (2024).