# Breakup dynamics of thin liquid sheets with viscous interfaces

### Highlights:
- Planar and axisymmetric linear stability analisys of thin liquid sheets;
- Effect of interface viscoelasticity on the breakup time of thin liquid sheets

### Abstract:
The interfacial shear rheological properties of thin liquid films are of outmost importance in understanding the mechanisms envolved in its dynamics. The stability and rupture of liquid films is of great importance in several applied sciences, whereas an initial disturbance to it can under certain conditions lead the liquid film  to break up. The stability of planar liquid sheet is driven by both van der Waals destructive effects and capillary and viscous stabilizing effects (Bazzi and Carvalho, 2018; Kistler and Scriven, 1984). In this paper we examine the nonlinear dynamics of a thin viscous film with viscous free-surface boundary conditions.

### Hypothesis
The main hypothesis are:
- Two-dimensional domain;
- Immiscible phases separated by a phase boundary $\Gamma (t)$ at which material parameters (density and absolute viscosity) have a jump-discontinuity;
- Isothermal flow;
- Newtonian and incompressible fluids;
- Well-posed problem (initial and boundary condition for the state parameters are known).

### Domain

- Let $\Omega \; \subset R^2$ be a domain containing two different immisciple incompressible phases. We denote the time-dependent subdomains containing the two phases as $\Omega_1(t)$ and $ \Omega_2(t) $, with $\bar{\Omega} = \bar{\Omega_1} \cup \bar{\Omega_2}$ and $\Omega_1 \cap \Omega_2 = 0$. 
- $\Omega_1$ and $\Omega_2$ are connected and $\partial \Omega_1 \cap \partial \Omega = 0$ (meaning $\Omega_1$ is completely contained in $\Omega$).
- The interface between $ \Omega_1$ and $ \Omega_2$ is denoted by $ \Gamma(t) =  \bar{\Omega_1}(t) \cup \bar{\Omega_2}(t)$.

### Bulk dynamics
The bulk stress tensor is given by:
$$ \sigma = -pI + \mu D(u), \; \; \; D(u) = \nabla u + (\nabla u)^T \; \; \;$$
With $p = p(x, t)$ as the pressure term, $u = u(x,t)$ as the velocity and $\mu$ as the bulk viscosity (assumed to be constant in each of the two phases). The bulk dynamics are provided by the conservation laws for mass and momentum under the assumption that the liquid sheet is bounded by a passive gas:

$$ \rho_i (\frac{\partial u}{\partial t} + (u \cdot \nabla)u) = -\nabla p + \nabla \cdot (\mu_i [\nabla u + (\nabla u)^T]) \;\;\;\; in \;\; \Omega_i \times [0,T]$$
$$ \nabla \cdot u = 0 \;\;\;\; in \;\; \Omega_i \times [0,T]$$
$$ i = 1,2. $$ 

Where the index $i$ associate the material parameters with the respective fluid phase. 

### Interface dynamics
When the interface is deformed, several phenomena contribute for the rising of different stresses at the interface. The constitutive model for the interface stress is given as (Hermans et al., 2015): 
$$ \sigma = \sigma_{\alpha\beta}(\Gamma)I + \sigma_e$$
$\alpha$ and $\beta$ are the fluid and gas phases, $\sigma_{\alpha\beta}(\Gamma)$ is the scalar surface tension between $\alpha$ and $\beta$, $I$ is a $2 \times 2$ identity matrix, $\Gamma$ is the surface concentration (not to be confused with the interface position) and $\sigma_e$ is the deviatoric surface stress tensor (also related to intrinsic rheological material functions). 

The surface tension will depend on the concentration of the surface active species, therefore the surface tension may vary due to adsorption-desorption kinetics associated with the change in interface area. However, Marangoni stresses arise with spatial variations in surface tension. Also, changes in interface curvature lead to pressure gradients due to capillary effects. 

Boussinesq (1913) introduced rheological material functions to explain observed phenomena of small drops and bubbles translation through a liquid. Later in the 1960s, Scriven (1960) introduced the constitutive equation for a Newtonian interface, known as the Boussinesq-Scriven constitutive equation. The constitutive law models interfaces with a viscous response.


By introducing the projection operator $P = I - nn^T$, with $n = n_\Gamma$ as the unit normal at the interface $\Gamma$ (pointing from $\Omega_1$ into $\Omega_2$), the surface deformation tensor is given by

$$ D_\Gamma (u) := P(\nabla_\Gamma u + (\nabla_\Gamma u)^T)P $$

With $\nabla_\Gamma$ the surface gradient. The Boussinesq-Scriven constitutive law computes the surface stress tensor $\sigma_\Gamma$ by cinsidering a surface ** dilatational viscosity ** coefficient $\lambda_\Gamma$ and a surface ** shear viscosity ** coefficient $\mu_\Gamma$. It is assumed that $\lambda_\Gamma \ge \mu_\Gamma \ge 0$. The surface stress tensor is written as 

$$ \sigma_\Gamma = [\tau + (\lambda_\Gamma - \mu_\Gamma) \; \nabla_\Gamma \cdot u]P + \mu_\Gamma D_\Gamma (u)$$ 

In which $\tau$ is the interfacial tension. For 2D flows, the constitutive model for the surface stress is simplified as below (REF - dissertação do Paulo Roberto). 
$$ \sigma_\Gamma = [\tau + (\lambda_\Gamma - \mu_\Gamma) \; \nabla_\Gamma \cdot u]P $$
The equation above shows that if the interface viscosity coefficients are equal to zero, the model recovers isotropic behavior for the stress balance on the interface.

### Van der Waals effects

Van der Waals potential $\Phi$ is given as
$$ \Phi = \frac{A}{2 \pi h^3} $$
Where A is the Hamacker constant. 

*section unfinished*

### Balance of forces on the interface

The balance of forces along the interface must take into consideration the deviatoric effects from the viscous response of the interface, capillary effects and the van der Waals effects from the liquid sheet thickness. Thus, we obtain
$$ n\cdot T = -p n + \nabla_\Gamma \cdot \sigma_\Gamma + \nabla \Phi $$

### Model

The standard model is based on the conservation laws for mass and momentum and  the Boussinesq-Scriven constitutive law for describing an interface dynamics. Moreover, the effects of ** van der Waals ** forces are also considered on the interface to account for the thickness of the liquid sheet. The boundary and initial condition are also explicited below.

- Bulk:
$$ \rho_i (\frac{\partial u}{\partial t} + (u \cdot \nabla)u) = -\nabla p + \nabla \cdot (\mu_i [\nabla u + (\nabla u)^T]) \;\;\;\; in \;\; \Omega_i \times [0,T]$$
$$ \nabla \cdot u = 0 \;\;\;\; in \;\; \Omega_i \times [0,T]$$
$$ i = 1,2. $$


- Boundary conditions:

    - Left and right - **zero gradient**: The velocity does not change in the normal direction regarding the boundary.
    $$ n \cdot \nabla u = 0 $$
    - Bottom -  ** symmetry line **: Absence of flux through the boundary and no tangential efforts.
    $$ n \cdot u = 0 $$ 
    $$n \cdot T \cdot t = 0 $$
    - Free surface (top) - ** *Boussinesq-Scriven constitutive equation* ** and ** *van der Waals* ** effects. Also, there is no flux across the interface (immiscible phases). 
    $$ n \cdot T = -p n + \nabla_\Gamma \cdot \sigma_\Gamma + \nabla \Phi  $$
    $$ n \cdot v = 0 $$
        
#### Check references

### Comparative literature:
- Erneux and Davis (1993): 
- Ida and Miksis (1995):
- Bazzi and Carvalho (2019):

### References

- Literature review:

Boussinesq J. 1913. Existence of a superficial viscosity in the thin transition layer separating one liquid from another contiguous fluid. C. R. Acad. Sci. 156:983–89

Kistler, S.F., Scriven, L.E. (1984). Coating flow theory by finite element and asymptotic analysis of the navier-stokes system. International Journal for Numerical Methods in Fluids 4 (3) 207-229.

C. Marangoni. Sull’espansione delle goccie d’un liquido galleggianti sulla superfice di altro liquido. Fratelli Fusi, 1865.

Scriven LE. 1960. Dynamics of a fluid interface. Equation ofmotion for Newtonian surface. Chem. Eng. Sci. 12:98–108

- Numerical validation:

Bazzi, M.S., Carvalho, M.S. (2018). Effect of viscoelasticity on liquid sheet rupture, Journal of Non-Newtonian Fluid Mechanics. DOI: https://doi.org/10.1016/j.jnnfm.2018.10.007


- Numerical modeling:

Vaynblat, D., Lister, J.R. (2001). Rupture of thin viscous films by van der Waals forces: Evolution and self-similarity. Phys. Fluids, Vol. 13, No. 5. DOI: 10.1063/1.1359749

Reusken, A., Zhang, Y. (2013). Numerical simulation of incompressible two-phase flows with a Boussinesq-Scriven interface stress tensor. Int. j. Numr. Meth. Fluids. DOI: 10.1002/fld.3835

