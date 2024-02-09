#pragma once

#include <eigen3/Eigen/Sparse>
#include <optional>

namespace EiCOS
{

    enum class exitcode
    {
        optimal = 0,           /* Problem solved to optimality              */
        primal_infeasible = 1, /* Found certificate of primal infeasibility */
        dual_infeasible = 2,   /* Found certificate of dual infeasibility   */
        maxit = -1,            /* Maximum number of iterations reached      */
        numerics = -2,         /* Search direction unreliable               */
        outcone = -3,          /* s or z got outside the cone, numerics?    */
        fatal = -7,            /* Unknown problem in solver                 */
        close_to_optimal = 10,
        close_to_primal_infeasible = 11,
        close_to_dual_infeasible = 12,
        not_converged_yet = -87
    };

    struct Settings
    {
        const long double gamma = 0.99L;         // scaling the final step length
        const long double delta = 2e-7L;         // regularization parameter
        const long double deltastat = 7e-8L;     // static regularization parameter
        const long double eps = 1e13L;           // regularization threshold
        const long double feastol = 1e-9L;       // primal/dual infeasibility tolerance
        const long double abstol = 1e-9L;        // absolute tolerance on duality gap
        const long double reltol = 1e-9L;        // relative tolerance on duality gap
        const long double feastol_inacc = 1e-4L; // primal/dual infeasibility relaxed tolerance
        const long double abstol_inacc = 5e-5L;  // absolute relaxed tolerance on duality gap
        const long double reltol_inacc = 5e-5L;  // relative relaxed tolerance on duality gap
        const size_t nitref = 9;           // maximum number of iterative refinement steps
        const size_t maxit = 100;          // maximum number of iterations
        bool verbose = false;              // print solver output
        const long double linsysacc = 1e-14L;    // rel. accuracy of search direction
        const long double irerrfact = 6L;        // factor by which IR should reduce err
        const long double stepmin = 1e-6L;       // smallest step that we do take
        const long double stepmax = 0.999L;      // largest step allowed, also in affine dir.
        const long double sigmamin = 1e-4L;      // always do some centering
        const long double sigmamax = 1.L;        // never fully center
        const size_t equil_iters = 3;      // eqilibration iterations
        const size_t iter_max = 100;       // maximum solver iterations
        const size_t safeguard = 500;      // Maximum increase in PRES before NUMERICS is thrown.
    };

    struct Information
    {
        long double pcost;
        long double dcost;
        long double pres;
        long double dres;
        bool pinf;
        bool dinf;
        std::optional<long double> pinfres;
        std::optional<long double> dinfres;
        long double gap;
        std::optional<long double> relgap;
        long double sigma;
        long double mu;
        long double step;
        long double step_aff;
        long double kapovert;
        size_t iter;
        size_t iter_max;
        size_t nitref1;
        size_t nitref2;
        size_t nitref3;

        bool isBetterThan(Information &other) const;
    };

    struct LPCone
    {
        Eigen::Vector<long double, Eigen::Dynamic>  w; // size n_lc
        Eigen::Vector<long double, Eigen::Dynamic>  v; // size n_lc
    };

    struct SOCone
    {
        size_t dim;            // dimension of cone
        Eigen::Vector<long double, Eigen::Dynamic>  skbar; // temporary variables to work with
        Eigen::Vector<long double, Eigen::Dynamic>  zkbar; // temporary variables to work with
        long double a;              // = wbar(1)
        long double d1;             // first element of D
        long double w;              // = q'*q
        long double eta;            // eta = (sres / zres)^(1/4)
        long double eta_square;     // eta^2 = (sres / zres)^(1/2)
        Eigen::Vector<long double, Eigen::Dynamic>  q;     // = wbar(2:end)
        long double u0;             // eta
        long double u1;             // u = [u0; u1 * q]
        long double v1;             // v = [0; v1 * q]
    };

    struct Work
    {
        void allocate(size_t n_var, size_t n_eq, size_t n_ineq);
        Eigen::Vector<long double, Eigen::Dynamic>  x;      // Primal variables  size n_var
        Eigen::Vector<long double, Eigen::Dynamic>  y;      // Multipliers for equality constaints  (size n_eq)
        Eigen::Vector<long double, Eigen::Dynamic>  z;      // Multipliers for conic inequalities   (size n_ineq)
        Eigen::Vector<long double, Eigen::Dynamic>  s;      // Slacks for conic inequalities        (size n_ineq)
        Eigen::Vector<long double, Eigen::Dynamic>  lambda; // Scaled variable                      (size n_ineq)

        // Homogeneous embedding
        long double kap; // kappa
        long double tau; // tau

        // Temporary storage
        long double cx, by, hz;

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
     *         \ / ########  ##  ##        ##    ##  ########
     *         / \ ########  ##  ##        ##    ##  ########
     *        /    ##        ##  ##        ##    ##        ##
     *       /     ########  ##  ########  ########  ########
     *      /      ########  ##  ########  ########  ########
     *     /          \
     *    /            \
     *    `'---....---'´
     *
     */

    public:
        Solver(const Eigen::SparseMatrix<long double> &G,
               const Eigen::SparseMatrix<long double> &A,
               const Eigen::Vector<long double, Eigen::Dynamic>  &c,
               const Eigen::Vector<long double, Eigen::Dynamic>  &h,
               const Eigen::Vector<long double, Eigen::Dynamic>  &b,
               const Eigen::VectorXi &soc_dims);
        void updateData(const Eigen::SparseMatrix<long double> &G,
                        const Eigen::SparseMatrix<long double> &A,
                        const Eigen::Vector<long double, Eigen::Dynamic>  &c,
                        const Eigen::Vector<long double, Eigen::Dynamic>  &h,
                        const Eigen::Vector<long double, Eigen::Dynamic>  &b);

        // traditional interface for compatibility
        Solver(int n, int m, int p, int l, int ncones, int *q,
               long double *Gpr, int *Gjc, int *Gir,
               long double *Apr, int *Ajc, int *Air,
               long double *c, long double *h, long double *b);
        void updateData(long double *Gpr, long double *Apr,
                        long double *c, long double *h, long double *b);

        exitcode solve(bool verbose = false);

        const Eigen::Vector<long double, Eigen::Dynamic>  &solution() const;

        Settings &getSettings();
        const Information &getInfo() const;

        // void saveProblemData(const std::string &path = "problem_data.hpp");

    private:
        void build(const Eigen::SparseMatrix<long double> &G,
                   const Eigen::SparseMatrix<long double> &A,
                   const Eigen::Vector<long double, Eigen::Dynamic>  &c,
                   const Eigen::Vector<long double, Eigen::Dynamic>  &h,
                   const Eigen::Vector<long double, Eigen::Dynamic>  &b,
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

        Eigen::SparseMatrix<long double> G;
        Eigen::SparseMatrix<long double> A;
        Eigen::SparseMatrix<long double> Gt;
        Eigen::SparseMatrix<long double> At;
        Eigen::Vector<long double, Eigen::Dynamic>  c;
        Eigen::Vector<long double, Eigen::Dynamic>  h;
        Eigen::Vector<long double, Eigen::Dynamic>  b;

        // Residuals
        Eigen::Vector<long double, Eigen::Dynamic>  rx; // (size n_var)
        Eigen::Vector<long double, Eigen::Dynamic>  ry; // (size n_eq)
        Eigen::Vector<long double, Eigen::Dynamic>  rz; // (size n_ineq)
        long double hresx, hresy, hresz;
        long double rt;

        // Norm iterates
        long double nx, ny, nz, ns;

        // Equilibration vectors
        Eigen::Vector<long double, Eigen::Dynamic>  x_equil; // (size n_var)
        Eigen::Vector<long double, Eigen::Dynamic>  A_equil; // (size n_eq)
        Eigen::Vector<long double, Eigen::Dynamic>  G_equil; // (size n_ineq)
        bool equibrilated;

        // The problem data scaling parameters
        long double resx0, resy0, resz0;

        Eigen::Vector<long double, Eigen::Dynamic>  dsaff_by_W, W_times_dzaff, dsaff;

        // KKT
        Eigen::Vector<long double, Eigen::Dynamic>  rhs1; // The right hand side in the first  KKT equation.
        Eigen::Vector<long double, Eigen::Dynamic>  rhs2; // The right hand side in the second KKT equation.
        Eigen::SparseMatrix<long double> K;
        using LDLT_t = Eigen::SimplicialLDLT<Eigen::SparseMatrix<long double>, Eigen::Upper>;
        LDLT_t ldlt;
        std::vector<long double *> KKT_V_ptr;  // Pointer to scaling/regularization elements for fast update
        std::vector<long double *> KKT_AG_ptr; // Pointer to A/G elements for fast update
        void setupKKT();
        void resetKKTScalings();
        void updateKKTScalings();
        void updateKKTAG();
        size_t solveKKT(const Eigen::Vector<long double, Eigen::Dynamic>  &rhs,
                        Eigen::Vector<long double, Eigen::Dynamic>  &dx,
                        Eigen::Vector<long double, Eigen::Dynamic>  &dy,
                        Eigen::Vector<long double, Eigen::Dynamic>  &dz,
                        bool initialize);

        void allocate();

        void bringToCone(const Eigen::Vector<long double, Eigen::Dynamic>  &r, Eigen::Vector<long double, Eigen::Dynamic>  &s);
        void computeResiduals();
        void updateStatistics();
        exitcode checkExitConditions(bool reduced_accuracy);
        bool updateScalings(const Eigen::Vector<long double, Eigen::Dynamic>  &s,
                            const Eigen::Vector<long double, Eigen::Dynamic>  &z,
                            Eigen::Vector<long double, Eigen::Dynamic>  &lambda);
        void RHSaffine();
        void RHScombined();
        void scale2add(const Eigen::Vector<long double, Eigen::Dynamic>  &x, Eigen::Vector<long double, Eigen::Dynamic>  &y);
        void scale(const Eigen::Vector<long double, Eigen::Dynamic>  &z, Eigen::Vector<long double, Eigen::Dynamic>  &lambda);
        long double lineSearch(Eigen::Vector<long double, Eigen::Dynamic>  &lambda,
                          Eigen::Vector<long double, Eigen::Dynamic>  &ds,
                          Eigen::Vector<long double, Eigen::Dynamic>  &dz,
                          long double tau,
                          long double dtau,
                          long double kap,
                          long double dkap);
        long double conicProduct(const Eigen::Vector<long double, Eigen::Dynamic>  &u,
                            const Eigen::Vector<long double, Eigen::Dynamic>  &v,
                            Eigen::Vector<long double, Eigen::Dynamic>  &w);
        void conicDivision(const Eigen::Vector<long double, Eigen::Dynamic>  &u,
                           const Eigen::Vector<long double, Eigen::Dynamic>  &w,
                           Eigen::Vector<long double, Eigen::Dynamic>  &v);
        void backscale();
        void setEquilibration();
        void unsetEquilibration();
        void cacheIndices();
        void printSummary();
    };

} // namespace EiCOS
