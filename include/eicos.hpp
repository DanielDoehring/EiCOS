#pragma once

#include <eigen3/Eigen/Sparse>
#include <optional>

#include <boost/multiprecision/cpp_dec_float.hpp> 
#include <boost/multiprecision/eigen.hpp>

//using float_type = double;
using float_type = boost::multiprecision::cpp_dec_float_50;

namespace EiCOS
{

    enum class exitcode
    {
        optimal = 0,           /* Problem solved to optimality               */
        primal_infeasible = 1, /* Found certificate of primal infeasibility  */
        dual_infeasible = 2,   /* Found certificate of dual infeasibility    */
        maxit = -1,            /* Maximum number of iterations reached       */
        numerics = -2,         /* Search direction unreliable                */
        outcone = -3,          /* s or z got outside the cone, numerics?     */
        fatal = -7,            /* Unknown problem in solver                  */
        close_to_optimal = 10,
        close_to_primal_infeasible = 11,
        close_to_dual_infeasible = 12,
        not_converged_yet = -87
    };

    struct Settings
    {
        const float_type gamma = 0.99;         // scaling the final step length
        const float_type delta = 2e-7;         // regularization parameter
        const float_type deltastat = 7e-8;     // static regularization parameter
        const float_type eps = 1e13;           // regularization threshold

        // TODO: These have always to be adjusted for more stages ...
        const float_type feastol = 1e-15;       // primal/dual infeasibility tolerance
        const float_type abstol = 1e-15;        // absolute tolerance on duality gap
        const float_type reltol = 1e-15;        // relative tolerance on duality gap

        const float_type feastol_inacc = 1e-4; // primal/dual infeasibility relaxed tolerance
        const float_type abstol_inacc = 5e-5;  // absolute relaxed tolerance on duality gap
        const float_type reltol_inacc = 5e-5;  // relative relaxed tolerance on duality gap
        const size_t nitref = 9;           // maximum number of iterative refinement steps
        const size_t maxit = 100;          // maximum number of iterations
        bool verbose = false;              // print solver output
        const float_type linsysacc = 1e-14;    // rel. accuracy of search direction
        const float_type irerrfact = 6;        // factor by which IR should reduce err
        const float_type stepmin = 1e-6;       // smallest step that we do take
        const float_type stepmax = 0.999;      // largest step allowed, also in affine dir.
        const float_type sigmamin = 1e-4;      // always do some centering
        const float_type sigmamax = 1.;        // never fully center
        const size_t equil_iters = 3;      // eqilibration iterations
        const size_t iter_max = 100;       // maximum solver iterations
        const size_t safeguard = 500;      // Maximum increase in PRES before NUMERICS is thrown.
    };

    struct Information
    {
        float_type pcost;
        float_type dcost;
        float_type pres;
        float_type dres;
        bool pinf;
        bool dinf;
        std::optional<float_type> pinfres;
        std::optional<float_type> dinfres;
        float_type gap;
        std::optional<float_type> relgap;
        float_type sigma;
        float_type mu;
        float_type step;
        float_type step_aff;
        float_type kapovert;
        size_t iter;
        size_t iter_max;
        size_t nitref1;
        size_t nitref2;
        size_t nitref3;

        bool isBetterThan(Information &other) const;
    };

    struct LPCone
    {
        Eigen::Vector<float_type, Eigen::Dynamic>  w; // size n_lc
        Eigen::Vector<float_type, Eigen::Dynamic>  v; // size n_lc
    };

    struct SOCone
    {
        size_t dim;            // dimension of cone
        Eigen::Vector<float_type, Eigen::Dynamic>  skbar; // temporary variables to work with
        Eigen::Vector<float_type, Eigen::Dynamic>  zkbar; // temporary variables to work with
        float_type a;              // = wbar(1)
        float_type d1;             // first element of D
        float_type w;              // = q'*q
        float_type eta;            // eta = (sres / zres)^(1/4)
        float_type eta_square;     // eta^2 = (sres / zres)^(1/2)
        Eigen::Vector<float_type, Eigen::Dynamic>  q;     // = wbar(2:end)
        float_type u0;             // eta
        float_type u1;             // u = [u0; u1 * q]
        float_type v1;             // v = [0; v1 * q]
    };

    struct Work
    {
        void allocate(const size_t n_var, const size_t n_eq, const size_t n_ineq);
        Eigen::Vector<float_type, Eigen::Dynamic>  x;      // Primal variables  size n_var
        Eigen::Vector<float_type, Eigen::Dynamic>  y;      // Multipliers for equality constaints  (size n_eq)
        Eigen::Vector<float_type, Eigen::Dynamic>  z;      // Multipliers for conic inequalities   (size n_ineq)
        Eigen::Vector<float_type, Eigen::Dynamic>  s;      // Slacks for conic inequalities        (size n_ineq)
        Eigen::Vector<float_type, Eigen::Dynamic>  lambda; // Scaled variable                      (size n_ineq)

        // Homogeneous embedding
        float_type kap; // kappa
        float_type tau; // tau

        // Temporary storage
        float_type cx, by, hz;

        Information i;
    };

    class Solver
    {
    /**
      *
      *    ..---''''---..
      *    \ '''----''' /
      *     \          /
      *      \      ########  ##  ########  ########  ########
      *       \     ########  ##  ########  ########  ########
      *        \    ##            ##        ##    ##  ##
      *          \/ ########  ##  ##        ##    ##  ########
      *          /\ ########  ##  ##        ##    ##  ########
      *        /    ##        ##  ##        ##    ##        ##
      *       /     ########  ##  ########  ########  ########
      *      /      ########  ##  ########  ########  ########
      *     /          \
      *    /            \
      *    `'---....---'´
      *
      */

    public:
        Solver(const Eigen::SparseMatrix<float_type> &G,
               const Eigen::SparseMatrix<float_type> &A,
               const Eigen::Vector<float_type, Eigen::Dynamic>  &c,
               const Eigen::Vector<float_type, Eigen::Dynamic>  &h,
               const Eigen::Vector<float_type, Eigen::Dynamic>  &b,
               const Eigen::VectorXi &soc_dims);
        void updateData(const Eigen::SparseMatrix<float_type> &G,
                        const Eigen::SparseMatrix<float_type> &A,
                        const Eigen::Vector<float_type, Eigen::Dynamic>  &c,
                        const Eigen::Vector<float_type, Eigen::Dynamic>  &h,
                        const Eigen::Vector<float_type, Eigen::Dynamic>  &b);

        // traditional interface for compatibility
        Solver(const int n, const int m, const int p, const int l, const int ncones, int *q,
               float_type *Gpr, int *Gjc, int *Gir,
               float_type *Apr, int *Ajc, int *Air,
               float_type *c, float_type *h, float_type *b);
        void updateData(float_type *Gpr, float_type *Apr,
                        float_type *c, float_type *h, float_type *b);

        exitcode solve(bool verbose = false);

        const Eigen::Vector<float_type, Eigen::Dynamic>  &solution() const;

        Settings &getSettings();
        const Information &getInfo() const;

        // void saveProblemData(const std::string &path = "problem_data.hpp");

    private:
        void build(const Eigen::SparseMatrix<float_type> &G,
                   const Eigen::SparseMatrix<float_type> &A,
                   const Eigen::Vector<float_type, Eigen::Dynamic>  &c,
                   const Eigen::Vector<float_type, Eigen::Dynamic>  &h,
                   const Eigen::Vector<float_type, Eigen::Dynamic>  &b,
                   const Eigen::VectorXi &soc_dims);

        Settings settings;
        Work w, w_best;

        size_t n_var;  // Number of variables (n)
        size_t n_eq;   // Number of equality constraints (p)
        size_t n_ineq; // Number of inequality constraints (m)
        size_t n_lc;   // Number of linear constraints (l)
        size_t n_sc;   // Number of second order cone constraints (ncones)
        size_t dim_K;  // Dimension of KKT matrix

        LPCone lp_cone;
        std::vector<SOCone> so_cones;

        Eigen::SparseMatrix<float_type> G;
        Eigen::SparseMatrix<float_type> A;
        Eigen::SparseMatrix<float_type> Gt;
        Eigen::SparseMatrix<float_type> At;
        Eigen::Vector<float_type, Eigen::Dynamic>  c;
        Eigen::Vector<float_type, Eigen::Dynamic>  h;
        Eigen::Vector<float_type, Eigen::Dynamic>  b;

        // Residuals
        Eigen::Vector<float_type, Eigen::Dynamic>  rx; // (size n_var)
        Eigen::Vector<float_type, Eigen::Dynamic>  ry; // (size n_eq)
        Eigen::Vector<float_type, Eigen::Dynamic>  rz; // (size n_ineq)
        float_type hresx, hresy, hresz;
        float_type rt;

        // Norm iterates
        float_type nx, ny, nz, ns;

        // Equilibration vectors
        Eigen::Vector<float_type, Eigen::Dynamic>  x_equil; // (size n_var)
        Eigen::Vector<float_type, Eigen::Dynamic>  A_equil; // (size n_eq)
        Eigen::Vector<float_type, Eigen::Dynamic>  G_equil; // (size n_ineq)
        bool equibrilated;

        // The problem data scaling parameters
        float_type resx0, resy0, resz0;

        Eigen::Vector<float_type, Eigen::Dynamic>  dsaff_by_W, W_times_dzaff, dsaff;

        // KKT
        Eigen::Vector<float_type, Eigen::Dynamic>  rhs1; // The right hand side in the first  KKT equation.
        Eigen::Vector<float_type, Eigen::Dynamic>  rhs2; // The right hand side in the second KKT equation.
        Eigen::SparseMatrix<float_type> K;
        using LDLT_t = Eigen::SimplicialLDLT<Eigen::SparseMatrix<float_type>, Eigen::Upper>;
        LDLT_t ldlt;
        std::vector<float_type *> KKT_V_ptr;  // Pointer to scaling/regularization elements for fast update
        std::vector<float_type *> KKT_AG_ptr; // Pointer to A/G elements for fast update
        void setupKKT();
        void resetKKTScalings();
        void updateKKTScalings();
        void updateKKTAG();
        size_t solveKKT(const Eigen::Vector<float_type, Eigen::Dynamic>  &rhs,
                        Eigen::Vector<float_type, Eigen::Dynamic>  &dx,
                        Eigen::Vector<float_type, Eigen::Dynamic>  &dy,
                        Eigen::Vector<float_type, Eigen::Dynamic>  &dz,
                        const bool initialize);

        void allocate();

        void bringToCone(const Eigen::Vector<float_type, Eigen::Dynamic>  &r, Eigen::Vector<float_type, Eigen::Dynamic>  &s);
        void computeResiduals();
        void updateStatistics();
        exitcode checkExitConditions(const bool reduced_accuracy);
        bool updateScalings(const Eigen::Vector<float_type, Eigen::Dynamic>  &s,
                            const Eigen::Vector<float_type, Eigen::Dynamic>  &z,
                            Eigen::Vector<float_type, Eigen::Dynamic>  &lambda);
        void RHSaffine();
        void RHScombined();
        void scale2add(const Eigen::Vector<float_type, Eigen::Dynamic>  &x, Eigen::Vector<float_type, Eigen::Dynamic>  &y);
        void scale(const Eigen::Vector<float_type, Eigen::Dynamic>  &z, Eigen::Vector<float_type, Eigen::Dynamic>  &lambda);
        float_type lineSearch(Eigen::Vector<float_type, Eigen::Dynamic>  &lambda,
                              Eigen::Vector<float_type, Eigen::Dynamic>  &ds,
                              Eigen::Vector<float_type, Eigen::Dynamic>  &dz,
                              const float_type tau, const float_type dtau,
                              const float_type kap, const float_type dkap);
        float_type conicProduct(const Eigen::Vector<float_type, Eigen::Dynamic>  &u,
                                const Eigen::Vector<float_type, Eigen::Dynamic>  &v,
                                Eigen::Vector<float_type, Eigen::Dynamic>  &w);
        void conicDivision(const Eigen::Vector<float_type, Eigen::Dynamic>  &u,
                           const Eigen::Vector<float_type, Eigen::Dynamic>  &w,
                           Eigen::Vector<float_type, Eigen::Dynamic>  &v);
        void backscale();
        void setEquilibration();
        void unsetEquilibration();
        void cacheIndices();
        void printSummary();
    };

} // namespace EiCOS
