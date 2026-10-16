// ============================================================================
//  pyu4lib.cpp  --  Python bindings for u4lib (pybind11).  Module name: u4lib.
//
//  Design goal: mirror the C++ API 1:1 so users who know u4lib use the same
//  names in Python. No pythonic renaming. u4lib is Wigner-Racah algebra for
//  U(4)/SU(4) (the Wigner supermultiplet symmetry). Coefficients come back
//  either as numpy arrays or as the Definitions.h container objects
//  (U4U3/U4SU2SU2/U4U6/U4Z6/U4U9), which give multiplicity-indexed __call__
//  and a flat .data() numpy view.
//
//  ==========================================================================
//  COMPLETE LIST OF EXPOSED NAMES  (Python side; C++ counterpart in u4::/su4::)
//  ==========================================================================
//
//  Module-level functions
//    initialize()                 set up foundation + u3 + U(4) DBs (call once)
//    finalize()                   tear everything down (call once at the end)
//    initialize_thread()          per-thread setup (advanced / multithreaded)
//    finalize_thread()            per-thread teardown
//    clear()                      clear cached coefficients (keep DBs alive)
//    is_initialized()             True only between initialize() and finalize()
//    Eta(P1a,P2a,P3a, ...b, ...c) coupling multiplicity, by SU(4) labels
//    Eta(a, b, c)                 coupling multiplicity, by irrep objects
//    decomposition(a, b)          output irreps and their Eta multiplicities
//    Zeta(P1, P2, P3, SS, TT)     (S,T) branching multiplicity, by labels
//    Zeta(irrep, SS, TT)          (S,T) branching multiplicity, by irrep object
//    physicalBasis(irrep)         list of physicalvector (zeta, SS, TT) states
//    gelfandBasis(irrep)          list of gelfand states spanning the irrep
//    RME_S/RME_T/RME_U(bra, ket)  (matrix, bra_zeta, ket_zeta) RME of S/T/U
//    phys_to_cano(p, MMS, MMT, gArr)  (canoBasis, T) Gelfand expansion
//    phys_to_cano_allM(p, gArr)   same for all (MMS,MMT), nested [iS][iT]
//
//  physicalvector(irrep, SS, TT)  a physical (SS,TT)-reduced basis state
//    irr(), P1(), P2(), P3(), n14(), n24(), n34(), n44(),
//    SS(), TT(), zeta(), zetamax(), repr
//
//  Database singletons (the ONLY way to reach coefficients -- no free funcs)
//    wigU4U3db()     -> WigU4U3_db        u4u6db() -> U4U6_db
//    wigU4SU2SU2db() -> WigU4SU2SU2_db    u4z6db() -> U4Z6_db
//                                         u4u9db() -> U4U9_db
//
//  su4.irrep(P1, P2, P3)          SU(4) irrep       str -> "(P1,P2,P3)"
//    P1(), P2(), P3(), n14(), n24(), n34(), n44(), dim(), zero(),
//    C2(), C3(), C4(), conjugate(), conjugate_phase_power(), str, repr,
//    full ordering (== != < <= > >=) and hashable (usable as a dict key)
//
//  irrep(P1, P2, P3) | irrep(n1, n2, n3, n4)   U(4) irrep; derives from su4.irrep
//    (inherits every su4.irrep method above, and adds / overrides:)
//    n14(), n24(), n34(), n44(), quanta(), C1(), normalize(), zero(),
//    conjugate(),  str -> "[n14,n24,n34,n44]",  repr,
//    ordering + hash that compare n44 (not just the su4 base labels)
//
//  gelfand(n14,n24,n34,n44, n13,n23,n33, n12,n22, n11)   Gelfand-Tsetlin state
//    n14(), n24(), n34(), n44(), n13(), n23(), n33(), n12(), n22(), n11(),
//    check(), normalize(), P1(), P2(), P3(), Lambdax2(), MLambdax2(),
//    conjugate(), conjugate_phase(), ==, !=, str, repr, hashable
//    generator matrix elements (44):
//      e11..e44, e12, e21, e23_12, e23_22, e32_12, e32_22, e13_12, e13_22,
//      e31_12, e31_22, e34_13/23/33, e43_13/23/33, and the three-step
//      e24_/e42_/e14_/e41_ {13,23,33}_{12,22} elements
//
//  Coefficient containers (returned by the get_U4XX accessors below). Each has
//  size(), empty(), data() -> numpy, __getitem__(i) (flat), __call__(...)
//  (multiplicity-indexed), str; plus these shape getters:
//    U4U3       etamax(), rhomax();        __call__(rho, eta)
//    U4SU2SU2   zetaAmax(), zetaBmax(), zetaCmax(), etamax();
//                                          __call__(zetaA, zetaB, zetaC, eta)
//    U4U6       etaABmax(), etaAB_Cmax(), etaBCmax(), etaA_BCmax();
//                                          __call__(etaAB, etaAB_C, etaBC, etaA_BC)
//    U4Z6       rhoABmax(), rhoAB_Cmax(), rhoACmax(), rhoAC_Bmax();
//                                          __call__(etaAB, etaAB_C, etaAC, etaAC_B)
//    U4U9       etaACmax(), etaBDmax(), etaABCDmax(), etaABmax(), etaCDmax(),
//               etaACBDmax();
//                       __call__(etaAC, etaBD, etaABCD, etaAB, etaCD, etaACBD)
//
//  Database methods
//    WigU4U3_db.get(<7 U(4)>U(3) labels per irrep a/b/c: n14,n24,n34,n44,
//                   n13,n23,n33>)                         -> 2-D numpy (rho x eta)
//    WigU4U3_db.get_U4U3(<same 21 labels>)                -> U4U3 (prints like C++)
//    WigU4SU2SU2_db.get(P1a,P2a,P3a,SSa,TTa, ...b, ...c)  -> flat numpy
//    WigU4SU2SU2_db.get_U4SU2SU2(<same 15 args>)          -> U4SU2SU2
//    U4U6_db.get_U4U6(u4A,u4B,u4ABC,u4C, u4AB,etaABmax,etaAB_Cmax,
//                     u4BC,etaBCmax,etaA_BCmax)            -> U4U6
//    U4Z6_db.get_U4Z6(u4B,u4A,u4ABC,u4C, u4AB,etaABmax,etaAB_Cmax,
//                     u4AC,etaACmax,etaAC_Bmax)            -> U4Z6
//    U4U9_db.get_U4U9(u4A,u4B,u4C,u4D, u4AB,u4CD, u4AC,u4BD, u4ABCD,
//                     etaACmax,etaBDmax,etaABCDmax, etaABmax,etaCDmax,etaACBDmax)
//                                                          -> U4U9
// ============================================================================

#include <sstream>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>      // Eigen::MatrixXd <-> numpy (wigU4U3db.get)
#include <pybind11/numpy.h>      // py::array_t
#include <pybind11/operators.h>  // py::self == py::self
#include <pybind11/stl.h>        // std::vector <-> list (physical/gelfand bases)

#include <u4lib.h>            // umbrella: initialize/finalize (cascades u4 -> u3
                             // -> utils) + all coefficient DBs
#include <U4basis.h>         // Zeta, physicalvector, physical/gelfand bases
#include <U4decomposition.h> // U(4) tensor-product decompositions
#include <U4generatorsRME.h>      // RME_S, RME_T, RME_U (physical generator RMEs)
#include <U4SU2SU2_to_U4U3U2U1.h> // phys_to_cano, phys_to_cano_allM

namespace py = pybind11;
using namespace py::literals;

namespace
{
  // Copy a contiguous double range (std::vector or std::span) into an owning
  // numpy array. Owning matters: U4U9 hands back a std::span view into a
  // thread-local scratch buffer that the next DB call overwrites.
  template <typename Range>
  py::array_t<double> to_numpy(const Range& r)
  {
    return py::array_t<double>(static_cast<py::ssize_t>(std::size(r)),
                               std::data(r));
  }

  // __str__ from a type's C++ operator<<.
  template <typename T>
  std::string to_str(const T& x)
  {
    std::ostringstream os;
    os << x;
    return os.str();
  }

  // Guard for database access. Reaching a coefficient DB before u4::initialize()
  // (or after u4::finalize()) dereferences a null cache / scratch buffer and
  // segfaults the process -- fatal in a Jupyter kernel. Raise a Python exception
  // instead. u4::initialized_flag() is true only between initialize/finalize.
  void require_init()
  {
    if (!u4::initialized_flag())
      throw std::runtime_error(
          "u4lib is not initialized: call u4lib.initialize() before using a "
          "coefficient database, and do not use the databases after "
          "u4lib.finalize().");
  }
} // namespace

PYBIND11_MODULE(u4lib, m)
{
  m.doc() =
      "u4lib: U(4)/SU(4) (Wigner supermultiplet) coupling and recoupling "
      "coefficients.\nNames mirror the C++ API. Call u4lib.initialize() once "
      "before use.";

  // ---- lifecycle -----------------------------------------------------------
  //  initialize() cascades u4 -> u3 -> utils, so one call sets up the
  //  Wigner backend, SO3 caches, and the U(3) and U(4) coefficient databases.
  m.def("initialize", []() { u4::initialize(); },
        "Initialize u4lib (call once before any coefficient call).");
  m.def("finalize", []() { u4::finalize(); },
        "Finalize u4lib (call once when done).");
  m.def("initialize_thread", []() { u4::initialize_thread(); });
  m.def("finalize_thread", []() { u4::finalize_thread(); });
  m.def("clear", &u4::clear, "Clear all cached coefficients (keep DBs alive).");
  m.def("is_initialized", []() { return u4::is_initialized(); },
        "True between initialize() and finalize(). The coefficient databases "
        "may only be used while this is True.");

  // ==========================================================================
  //  su4::irrep  --  an SU(4) irrep (P1, P2, P3).  Exposed as u4lib.su4.irrep.
  // ==========================================================================
  auto su4 = m.def_submodule("su4", "SU(4) irreps (mirrors the su4:: namespace).");
  py::class_<su4::irrep>(su4, "irrep")
      .def(py::init<>())
      .def(py::init([](int P1, int P2, int P3) { return su4::irrep(P1, P2, P3); }),
           "P1"_a, "P2"_a, "P3"_a)
      .def("P1", &su4::irrep::P1)
      .def("P2", &su4::irrep::P2)
      .def("P3", &su4::irrep::P3)
      .def("n14", &su4::irrep::n14)
      .def("n24", &su4::irrep::n24)
      .def("n34", &su4::irrep::n34)
      .def("n44", &su4::irrep::n44)
      .def("dim", &su4::irrep::dim, "Dimension of the SU(4) irrep.")
      .def("zero", &su4::irrep::zero, "True for the scalar irrep (0,0,0).")
      .def("C2", &su4::irrep::C2, "Second-order Casimir.")
      .def("C3", &su4::irrep::C3, "Third-order Casimir.")
      .def("C4", &su4::irrep::C4, "Fourth-order Casimir.")
      .def("conjugate", &su4::irrep::conjugate)
      .def("conjugate_phase_power", &su4::irrep::conjugate_phase_power)
      .def(py::self == py::self)   // full ordering, from the C++ operator<=>
      .def(py::self != py::self)
      .def(py::self < py::self)
      .def(py::self <= py::self)
      .def(py::self > py::self)
      .def(py::self >= py::self)
      .def("__hash__",
           [](const su4::irrep& i) {
             return py::hash(py::make_tuple(i.P1(), i.P2(), i.P3()));
           })
      .def("__str__", &to_str<su4::irrep>)     // "(P1,P2,P3)"
      .def("__repr__", [](const su4::irrep& i) {
        return "su4.irrep(" + std::to_string(i.P1()) + "," +
               std::to_string(i.P2()) + "," + std::to_string(i.P3()) + ")";
      });

  // ==========================================================================
  //  u4::irrep  --  a U(4) irrep. Labelled by SU(4) content (P1,P2,P3) with
  //  n44 = 0, or by the four U(4) row labels (n1 >= n2 >= n3 >= n4).
  // ==========================================================================
  py::class_<u4::irrep, su4::irrep>(m, "irrep")
      .def(py::init<>())
      .def(py::init([](int P1, int P2, int P3) { return u4::irrep(P1, P2, P3); }),
           "P1"_a, "P2"_a, "P3"_a, "U(4) irrep from SU(4) labels (n44 = 0).")
      .def(py::init([](int n1, int n2, int n3, int n4) {
             return u4::irrep(n1, n2, n3, n4);
           }),
           "n1"_a, "n2"_a, "n3"_a, "n4"_a,
           "U(4) irrep from row labels n1 >= n2 >= n3 >= n4.")
      // n14/n24/n34/n44 shadow the su4 base (they include n44 here):
      .def("n14", &u4::irrep::n14)
      .def("n24", &u4::irrep::n24)
      .def("n34", &u4::irrep::n34)
      .def("n44", &u4::irrep::n44)
      .def("quanta", &u4::irrep::quanta, "Total number of oscillator quanta.")
      .def("C1", &u4::irrep::C1, "First-order Casimir (= total quanta).")
      .def("normalize", &u4::irrep::normalize, "Set n44 to 0 in place.")
      .def("zero", &u4::irrep::zero, "True for the scalar irrep [0,0,0,0].")
      .def("conjugate", &u4::irrep::conjugate)
      // u4-level comparisons (from u4::irrep::operator<=>): these compare n44,
      // unlike the inherited su4 == which only sees (P1, P2, P3).
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::self < py::self)
      .def(py::self <= py::self)
      .def(py::self > py::self)
      .def(py::self >= py::self)
      .def("__hash__",
           [](const u4::irrep& i) {
             return py::hash(
                 py::make_tuple(i.n14(), i.n24(), i.n34(), i.n44()));
           })
      .def("__str__", &to_str<u4::irrep>)      // "[n14,n24,n34,n44]"
      .def("__repr__", [](const u4::irrep& i) {
        return "irrep(" + std::to_string(i.n14()) + "," +
               std::to_string(i.n24()) + "," + std::to_string(i.n34()) + "," +
               std::to_string(i.n44()) + ")";
      });

  // ==========================================================================
  //  u4::gelfand  --  a U(4) Gelfand-Tsetlin basis state (10 labels).
  //  Carries the SU(2) subgroup labels and the U(4) generator matrix elements.
  // ==========================================================================
  py::class_<u4::gelfand>(m, "gelfand")
      .def(py::init([](int n14, int n24, int n34, int n44, int n13, int n23,
                       int n33, int n12, int n22, int n11) {
             return u4::gelfand(n14, n24, n34, n44, n13, n23, n33, n12, n22, n11);
           }),
           "n14"_a, "n24"_a, "n34"_a, "n44"_a, "n13"_a, "n23"_a, "n33"_a,
           "n12"_a, "n22"_a, "n11"_a)
      // labels (const getters, via lambdas to avoid the getter/setter overload):
      .def("n14", [](const u4::gelfand& g) { return g.n14(); })
      .def("n24", [](const u4::gelfand& g) { return g.n24(); })
      .def("n34", [](const u4::gelfand& g) { return g.n34(); })
      .def("n44", [](const u4::gelfand& g) { return g.n44(); })
      .def("n13", [](const u4::gelfand& g) { return g.n13(); })
      .def("n23", [](const u4::gelfand& g) { return g.n23(); })
      .def("n33", [](const u4::gelfand& g) { return g.n33(); })
      .def("n12", [](const u4::gelfand& g) { return g.n12(); })
      .def("n22", [](const u4::gelfand& g) { return g.n22(); })
      .def("n11", [](const u4::gelfand& g) { return g.n11(); })
      .def("check", &u4::gelfand::check, "Betweenness (validity) condition.")
      .def("normalize", &u4::gelfand::normalize)
      .def("P1", &u4::gelfand::P1)
      .def("P2", &u4::gelfand::P2)
      .def("P3", &u4::gelfand::P3)
      .def("Lambdax2", &u4::gelfand::Lambdax2, "2 * SU(2) angular momentum.")
      .def("MLambdax2", &u4::gelfand::MLambdax2, "2 * SU(2) projection.")
      .def("conjugate", &u4::gelfand::conjugate)
      .def("conjugate_phase", &u4::gelfand::conjugate_phase)
      // U(4) generator matrix elements on this state (44):
      .def("e11", &u4::gelfand::e11).def("e22", &u4::gelfand::e22)
      .def("e33", &u4::gelfand::e33).def("e44", &u4::gelfand::e44)
      .def("e12", &u4::gelfand::e12).def("e21", &u4::gelfand::e21)
      .def("e23_12", &u4::gelfand::e23_12).def("e23_22", &u4::gelfand::e23_22)
      .def("e32_12", &u4::gelfand::e32_12).def("e32_22", &u4::gelfand::e32_22)
      .def("e13_12", &u4::gelfand::e13_12).def("e13_22", &u4::gelfand::e13_22)
      .def("e31_12", &u4::gelfand::e31_12).def("e31_22", &u4::gelfand::e31_22)
      .def("e34_13", &u4::gelfand::e34_13).def("e34_23", &u4::gelfand::e34_23)
      .def("e34_33", &u4::gelfand::e34_33)
      .def("e43_13", &u4::gelfand::e43_13).def("e43_23", &u4::gelfand::e43_23)
      .def("e43_33", &u4::gelfand::e43_33)
      .def("e24_13_12", &u4::gelfand::e24_13_12)
      .def("e24_13_22", &u4::gelfand::e24_13_22)
      .def("e24_23_12", &u4::gelfand::e24_23_12)
      .def("e24_23_22", &u4::gelfand::e24_23_22)
      .def("e24_33_12", &u4::gelfand::e24_33_12)
      .def("e24_33_22", &u4::gelfand::e24_33_22)
      .def("e42_13_12", &u4::gelfand::e42_13_12)
      .def("e42_13_22", &u4::gelfand::e42_13_22)
      .def("e42_23_12", &u4::gelfand::e42_23_12)
      .def("e42_23_22", &u4::gelfand::e42_23_22)
      .def("e42_33_12", &u4::gelfand::e42_33_12)
      .def("e42_33_22", &u4::gelfand::e42_33_22)
      .def("e14_13_12", &u4::gelfand::e14_13_12)
      .def("e14_13_22", &u4::gelfand::e14_13_22)
      .def("e14_23_12", &u4::gelfand::e14_23_12)
      .def("e14_23_22", &u4::gelfand::e14_23_22)
      .def("e14_33_12", &u4::gelfand::e14_33_12)
      .def("e14_33_22", &u4::gelfand::e14_33_22)
      .def("e41_13_12", &u4::gelfand::e41_13_12)
      .def("e41_13_22", &u4::gelfand::e41_13_22)
      .def("e41_23_12", &u4::gelfand::e41_23_12)
      .def("e41_23_22", &u4::gelfand::e41_23_22)
      .def("e41_33_12", &u4::gelfand::e41_33_12)
      .def("e41_33_22", &u4::gelfand::e41_33_22)
      .def(py::self == py::self)   // gelfand has == / != only (no C++ <=>)
      .def(py::self != py::self)
      .def("__hash__",
           [](const u4::gelfand& g) {
             return py::hash(py::make_tuple(g.n14(), g.n24(), g.n34(), g.n44(),
                                            g.n13(), g.n23(), g.n33(), g.n12(),
                                            g.n22(), g.n11()));
           })
      .def("__str__", &to_str<u4::gelfand>)
      .def("__repr__",
           [](const u4::gelfand& g) { return "gelfand(" + to_str(g) + ")"; });

  // ---- Eta : U(4) coupling multiplicity of (a) x (b) -> (c) ----------------
  m.def("Eta",
        [](int P1a, int P2a, int P3a, int P1b, int P2b, int P3b, int P1c,
           int P2c, int P3c) {
          return u4::Eta(u4::irrep(P1a, P2a, P3a), u4::irrep(P1b, P2b, P3b),
                         u4::irrep(P1c, P2c, P3c));
        },
        "P1a"_a, "P2a"_a, "P3a"_a, "P1b"_a, "P2b"_a, "P3b"_a, "P1c"_a, "P2c"_a,
        "P3c"_a, "Coupling multiplicity eta (by SU(4) labels).");
  m.def("Eta",
        [](const u4::irrep& a, const u4::irrep& b, const u4::irrep& c) {
          return u4::Eta(a, b, c);
        },
        "a"_a, "b"_a, "c"_a, "Coupling multiplicity eta (by irrep objects).");

  m.def("decomposition",
        [](const u4::irrep& a, const u4::irrep& b) {
          return u4::decomposition(a, b);
        },
        "a"_a, "b"_a,
        "Return (output U(4) irreps, Eta multiplicities) for the product a x b.");

  // ---- Zeta : (S,T) branching multiplicity of (SS,TT) in the irrep ---------
  //  SS = 2*spin, TT = 2*isospin (doubled quantum numbers).
  m.def("Zeta",
        [](int P1, int P2, int P3, int SS, int TT) {
          return u4::Zeta(u4::irrep(P1, P2, P3), SS, TT);
        },
        "P1"_a, "P2"_a, "P3"_a, "SS"_a, "TT"_a,
        "(S,T) branching multiplicity of (2S,2T) in (P1,P2,P3).");
  m.def("Zeta",
        [](const u4::irrep& a, int SS, int TT) { return u4::Zeta(a, SS, TT); },
        "irrep"_a, "SS"_a, "TT"_a, "(S,T) branching multiplicity (by irrep).");

  // ---- physicalvector : a physical (SS,TT)-reduced basis state -------------
  py::class_<u4::physicalvector>(m, "physicalvector")
      .def(py::init([](const u4::irrep& irr, int SS, int TT) {
             return u4::physicalvector(irr, SS, TT);
           }),
           "irrep"_a, "SS"_a, "TT"_a)
      .def("irr", &u4::physicalvector::irr,
           py::return_value_policy::reference_internal, "The U(4) irrep.")
      .def("P1", &u4::physicalvector::P1)
      .def("P2", &u4::physicalvector::P2)
      .def("P3", &u4::physicalvector::P3)
      .def("n14", &u4::physicalvector::n14)
      .def("n24", &u4::physicalvector::n24)
      .def("n34", &u4::physicalvector::n34)
      .def("n44", &u4::physicalvector::n44)
      .def("SS", &u4::physicalvector::SS, "2 * spin.")
      .def("TT", &u4::physicalvector::TT, "2 * isospin.")
      .def("zeta", &u4::physicalvector::zeta, "(S,T) multiplicity.")
      .def("zetamax", &u4::physicalvector::zetamax)
      .def("__repr__", [](const u4::physicalvector& p) {
        return "physicalvector((" + std::to_string(p.P1()) + "," +
               std::to_string(p.P2()) + "," + std::to_string(p.P3()) +
               "), zeta=" + std::to_string(p.zeta()) +
               ", SS=" + std::to_string(p.SS()) +
               ", TT=" + std::to_string(p.TT()) + ")";
      });

  // ---- basis enumerators (return Python lists) -----------------------------
  m.def("physicalBasis", &u4::physicalBasis, "irrep"_a,
        "List of physicalvector states (zeta, SS, TT) spanning the irrep.");
  m.def("gelfandBasis", &u4::gelfandBasis, "irrep"_a,
        "List of gelfand states spanning the irrep.");

  // ---- physical generator reduced matrix elements (S, T and U) -------------
  //  <(P1,P2,P3) zeta' S' T' || G || (P1,P2,P3) zeta S T> for the spin (S),
  //  isospin (T) and Gamow-Teller-like (U) generators, bra and ket in the SAME
  //  irrep. Returns a tuple (RME, bra_zeta, ket_zeta): RME is a
  //  (bra.zeta() x ket.zeta()) matrix indexed by zero-based (S,T) multiplicity;
  //  bra_zeta/ket_zeta are the zero-based zeta labels (0..zeta-1) for the rows/columns.
  //  Requires an initialized session.
  m.def("RME_S",
        [](u4::physicalvector bra, u4::physicalvector ket) {
          require_init();
          std::vector<int> bra_zeta, ket_zeta;
          Eigen::MatrixXd rme = u4::RME_S(bra, ket, bra_zeta, ket_zeta);
          return py::make_tuple(rme, bra_zeta, ket_zeta);
        },
        "bra"_a, "ket"_a,
        "Reduced matrix element of S: (matrix, bra_zeta, ket_zeta).");
  m.def("RME_T",
        [](u4::physicalvector bra, u4::physicalvector ket) {
          require_init();
          std::vector<int> bra_zeta, ket_zeta;
          Eigen::MatrixXd rme = u4::RME_T(bra, ket, bra_zeta, ket_zeta);
          return py::make_tuple(rme, bra_zeta, ket_zeta);
        },
        "bra"_a, "ket"_a,
        "Reduced matrix element of T: (matrix, bra_zeta, ket_zeta).");
  m.def("RME_U",
        [](u4::physicalvector bra, u4::physicalvector ket) {
          require_init();
          std::vector<int> bra_zeta, ket_zeta;
          Eigen::MatrixXd rme = u4::RME_U(bra, ket, bra_zeta, ket_zeta);
          return py::make_tuple(rme, bra_zeta, ket_zeta);
        },
        "bra"_a, "ket"_a,
        "Reduced matrix element of U: (matrix, bra_zeta, ket_zeta).");

  // ---- physical -> canonical (Gelfand) expansion ---------------------------
  //  Express |(P1,P2,P3) zeta S T; MS MT> in the Gelfand basis.
  //  phys_to_cano(p, MMS, MMT, gArr) returns (canoBasis, T): canoBasis are
  //  indices into gArr of the Gelfand states at this (MMS,MMT), and column k of
  //  the T matrix is the k-th (S,T) multiplicity copy (zeta = k+1). MMS/MMT are
  //  the doubled projections (2*MS, 2*MT). phys_to_cano_allM returns the same
  //  for every (MMS,MMT) as nested lists indexed [iS][iT] with iS=(SS-MMS)/2,
  //  iT=(TT-MMT)/2. Requires an initialized session.
  m.def("phys_to_cano",
        [](const u4::physicalvector& p, int MMS, int MMT,
           const std::vector<u4::gelfand>& gArr) {
          require_init();
          return u4::phys_to_cano(p, MMS, MMT, gArr);
        },
        "p"_a, "MMS"_a, "MMT"_a, "gArr"_a,
        "Expand |... S T; MS MT> in the Gelfand basis: (canoBasis indices, T).");
  m.def("phys_to_cano_allM",
        [](const u4::physicalvector& p, const std::vector<u4::gelfand>& gArr) {
          require_init();
          return u4::phys_to_cano_allM(p, gArr);
        },
        "p"_a, "gArr"_a,
        "phys_to_cano for every (MMS,MMT) as nested lists [iS][iT].");

  // ==========================================================================
  //  Coefficient container classes (Definitions.h). Structured results of the
  //  get_U4XX() accessors: own a flat buffer, multiplicity-indexed __call__,
  //  flat __getitem__, and .data() as numpy.
  // ==========================================================================
  // U4U3: coefficients over (rho, eta). WigU4U3_db.get returns the same data as
  // a 2-D numpy array directly; this container is here for family consistency.
  using U4U3_t = u4::U4U3<u4::coeffs_t>;
  py::class_<U4U3_t>(m, "U4U3")
      .def("size", &U4U3_t::size)
      .def("empty", &U4U3_t::empty)
      .def("etamax", &U4U3_t::etamax)
      .def("rhomax", &U4U3_t::rhomax)
      .def("__call__",
           [](const U4U3_t& o, size_t rho, size_t eta) { return o(rho, eta); },
           "rho"_a, "eta"_a)
      .def("__getitem__", [](const U4U3_t& o, size_t i) { return o[i]; })
      .def("data", [](const U4U3_t& o) { return to_numpy(o.data()); })
      .def("print", [](const U4U3_t& o) { o.print(); }, "Print formatted coefficients.")
      .def("__str__", &to_str<U4U3_t>);

  using U4SU2SU2_t = u4::U4SU2SU2<u4::coeffs_t>;
  py::class_<U4SU2SU2_t>(m, "U4SU2SU2")
      .def("size", &U4SU2SU2_t::size)
      .def("empty", &U4SU2SU2_t::empty)
      .def("zetaAmax", &U4SU2SU2_t::zetaAmax)
      .def("zetaBmax", &U4SU2SU2_t::zetaBmax)
      .def("zetaCmax", &U4SU2SU2_t::zetaCmax)
      .def("etamax", &U4SU2SU2_t::etamax)
      .def("__call__",
           [](const U4SU2SU2_t& o, size_t z1, size_t z2, size_t z3, size_t eta) {
             return o(z1, z2, z3, eta);
           },
           "zetaA"_a, "zetaB"_a, "zetaC"_a, "eta"_a)
      .def("__getitem__", [](const U4SU2SU2_t& o, size_t i) { return o[i]; })
      .def("data", [](const U4SU2SU2_t& o) { return to_numpy(o.data()); })
      .def("print", [](const U4SU2SU2_t& o) { o.print(); }, "Print formatted coefficients.")
      .def("__str__", &to_str<U4SU2SU2_t>);

  using U4U6_t = u4::U4U6<u4::coeffs_t>;
  py::class_<U4U6_t>(m, "U4U6")
      .def("size", &U4U6_t::size)
      .def("empty", &U4U6_t::empty)
      .def("etaABmax", &U4U6_t::etaABmax)
      .def("etaAB_Cmax", &U4U6_t::etaAB_Cmax)
      .def("etaBCmax", &U4U6_t::etaBCmax)
      .def("etaA_BCmax", &U4U6_t::etaA_BCmax)
      .def("__call__",
           [](const U4U6_t& o, size_t etaAB, size_t etaAB_C, size_t etaBC,
              size_t etaA_BC) { return o(etaAB, etaAB_C, etaBC, etaA_BC); },
           "etaAB"_a, "etaAB_C"_a, "etaBC"_a, "etaA_BC"_a)
      .def("__getitem__", [](const U4U6_t& o, size_t i) { return o[i]; })
      .def("data", [](const U4U6_t& o) { return to_numpy(o.data()); })
      .def("print", [](const U4U6_t& o) { o.print(); }, "Print formatted coefficients.")
      .def("__str__", &to_str<U4U6_t>);

  using U4Z6_t = u4::U4Z6<u4::coeffs_t>;
  py::class_<U4Z6_t>(m, "U4Z6")
      .def("size", &U4Z6_t::size)
      .def("empty", &U4Z6_t::empty)
      .def("rhoABmax", &U4Z6_t::rhoABmax)
      .def("rhoAB_Cmax", &U4Z6_t::rhoAB_Cmax)
      .def("rhoACmax", &U4Z6_t::rhoACmax)
      .def("rhoAC_Bmax", &U4Z6_t::rhoAC_Bmax)
      .def("__call__",
           [](const U4Z6_t& o, size_t etaAB, size_t etaAB_C, size_t etaAC,
              size_t etaAC_B) { return o(etaAB, etaAB_C, etaAC, etaAC_B); },
           "etaAB"_a, "etaAB_C"_a, "etaAC"_a, "etaAC_B"_a)
      .def("__getitem__", [](const U4Z6_t& o, size_t i) { return o[i]; })
      .def("data", [](const U4Z6_t& o) { return to_numpy(o.data()); })
      .def("print", [](const U4Z6_t& o) { o.print(); }, "Print formatted coefficients.")
      .def("__str__", &to_str<U4Z6_t>);

  using U4U9_t = u4::U4U9<u4::coeffs_t>;
  py::class_<U4U9_t>(m, "U4U9")
      .def("size", &U4U9_t::size)
      .def("empty", &U4U9_t::empty)
      .def("etaACmax", &U4U9_t::etaACmax)
      .def("etaBDmax", &U4U9_t::etaBDmax)
      .def("etaABCDmax", &U4U9_t::etaABCDmax)
      .def("etaABmax", &U4U9_t::etaABmax)
      .def("etaCDmax", &U4U9_t::etaCDmax)
      .def("etaACBDmax", &U4U9_t::etaACBDmax)
      .def("__call__",
           [](const U4U9_t& o, size_t etaAC, size_t etaBD, size_t etaABCD,
              size_t etaAB, size_t etaCD, size_t etaACBD) {
             return o(etaAC, etaBD, etaABCD, etaAB, etaCD, etaACBD);
           },
           "etaAC"_a, "etaBD"_a, "etaABCD"_a, "etaAB"_a, "etaCD"_a, "etaACBD"_a)
      .def("__getitem__", [](const U4U9_t& o, size_t i) { return o[i]; })
      .def("data", [](const U4U9_t& o) { return to_numpy(o.data()); })
      .def("print", [](const U4U9_t& o) { o.print(); }, "Print formatted coefficients.")
      .def("__str__", &to_str<U4U9_t>);

  // ==========================================================================
  //  Coefficient databases (singletons). All coefficients are reached through
  //  these -- there are no standalone free functions in the public API.
  // ==========================================================================

  // ---- U(4) > U(3) reduced coupling coefficients ---------------------------
  //  get(...) takes, per irrep a/b/c: (n14, n24, n34, n44, n13, n23, n33).
  //  Returns a 2-D numpy array indexed [rho, eta].
  py::class_<u4::WigU4U3_db, std::unique_ptr<u4::WigU4U3_db, py::nodelete>>(
      m, "WigU4U3_db")
      .def("get",
           [](u4::WigU4U3_db& db, int n14a, int n24a, int n34a, int n44a,
              int n13a, int n23a, int n33a, int n14b, int n24b, int n34b,
              int n44b, int n13b, int n23b, int n33b, int n14c, int n24c,
              int n34c, int n44c, int n13c, int n23c, int n33c) {
             require_init();
             return db.get(n14a, n24a, n34a, n44a, n13a, n23a, n33a, n14b, n24b,
                           n34b, n44b, n13b, n23b, n33b, n14c, n24c, n34c, n44c,
                           n13c, n23c, n33c);
           },
           "n14a"_a, "n24a"_a, "n34a"_a, "n44a"_a, "n13a"_a, "n23a"_a, "n33a"_a,
           "n14b"_a, "n24b"_a, "n34b"_a, "n44b"_a, "n13b"_a, "n23b"_a, "n33b"_a,
           "n14c"_a, "n24c"_a, "n34c"_a, "n44c"_a, "n13c"_a, "n23c"_a, "n33c"_a,
           "Coefficient matrix over (rho, eta) as a 2-D numpy array.")
      // Structured U4U3 object: same coefficients, but indexed by (rho, eta),
      // with .data() and a __str__/print that reproduce the C++ U4U3 layout.
      .def(
          "get_U4U3",
          [](u4::WigU4U3_db& db, int n14a, int n24a, int n34a, int n44a,
             int n13a, int n23a, int n33a, int n14b, int n24b, int n34b,
             int n44b, int n13b, int n23b, int n33b, int n14c, int n24c,
             int n34c, int n44c, int n13c, int n23c, int n33c) {
            require_init();
            return db.get_U4U3(n14a, n24a, n34a, n44a, n13a, n23a, n33a, n14b,
                               n24b, n34b, n44b, n13b, n23b, n33b, n14c, n24c,
                               n34c, n44c, n13c, n23c, n33c);
          },
          "n14a"_a, "n24a"_a, "n34a"_a, "n44a"_a, "n13a"_a, "n23a"_a, "n33a"_a,
          "n14b"_a, "n24b"_a, "n34b"_a, "n44b"_a, "n13b"_a, "n23b"_a, "n33b"_a,
          "n14c"_a, "n24c"_a, "n34c"_a, "n44c"_a, "n13c"_a, "n23c"_a, "n33c"_a,
          "Structured U4U3 object (indexed by rho, eta; prints like C++).");
  m.def("wigU4U3db", &u4::wigU4U3db, py::return_value_policy::reference,
        "The global U(4)>U(3) coefficient database (singleton).");

  // ---- U(4) > SU(2)xSU(2) reduced coupling coefficients --------------------
  //  get(...) per irrep: (P1, P2, P3, SS, TT). Flat numpy, or the structured
  //  U4SU2SU2 object via get_U4SU2SU2(...) (indexed zetaA,zetaB,zetaC,eta).
  py::class_<u4::WigU4SU2SU2_db,
             std::unique_ptr<u4::WigU4SU2SU2_db, py::nodelete>>(
      m, "WigU4SU2SU2_db")
      .def(
          "get",
          [](u4::WigU4SU2SU2_db& db, int P1a, int P2a, int P3a, int SSa, int TTa,
             int P1b, int P2b, int P3b, int SSb, int TTb, int P1c, int P2c,
             int P3c, int SSc, int TTc) {
            require_init();
            return to_numpy(db.get(P1a, P2a, P3a, SSa, TTa, P1b, P2b, P3b, SSb,
                                   TTb, P1c, P2c, P3c, SSc, TTc));
          },
          "P1a"_a, "P2a"_a, "P3a"_a, "SSa"_a, "TTa"_a, "P1b"_a, "P2b"_a, "P3b"_a,
          "SSb"_a, "TTb"_a, "P1c"_a, "P2c"_a, "P3c"_a, "SSc"_a, "TTc"_a,
          "Coefficients as a flat numpy array.")
      .def("get_U4SU2SU2",
           [](u4::WigU4SU2SU2_db& db, int P1a, int P2a, int P3a, int SSa,
              int TTa, int P1b, int P2b, int P3b, int SSb, int TTb, int P1c,
              int P2c, int P3c, int SSc, int TTc) {
             require_init();
             return db.get_U4SU2SU2(P1a, P2a, P3a, SSa, TTa, P1b, P2b, P3b, SSb,
                                    TTb, P1c, P2c, P3c, SSc, TTc);
           },
           "P1a"_a, "P2a"_a, "P3a"_a, "SSa"_a, "TTa"_a, "P1b"_a, "P2b"_a,
           "P3b"_a, "SSb"_a, "TTb"_a, "P1c"_a, "P2c"_a, "P3c"_a, "SSc"_a,
           "TTc"_a, "Structured U4SU2SU2 object (zetaA,zetaB,zetaC,eta).");
  m.def("wigU4SU2SU2db", &u4::wigU4SU2SU2db, py::return_value_policy::reference,
        "The global U(4)>SU(2)xSU(2) coefficient database (singleton).");

  // ---- U(4) 6-irrep recoupling: U-coefficients (U4U6) ----------------------
  py::class_<u4::U4U6_db, std::unique_ptr<u4::U4U6_db, py::nodelete>>(
      m, "U4U6_db")
      .def("get_U4U6",
           [](u4::U4U6_db& db, const u4::irrep& u4A, const u4::irrep& u4B,
              const u4::irrep& u4ABC, const u4::irrep& u4C, const u4::irrep& u4AB,
              int etaABmax, int etaAB_Cmax, const u4::irrep& u4BC, int etaBCmax,
              int etaA_BCmax) {
             require_init();
             return db.get_U4U6(u4A, u4B, u4ABC, u4C, u4AB, etaABmax, etaAB_Cmax,
                                u4BC, etaBCmax, etaA_BCmax);
           },
           "u4A"_a, "u4B"_a, "u4ABC"_a, "u4C"_a,
           "u4AB"_a, "etaABmax"_a, "etaAB_Cmax"_a,
           "u4BC"_a, "etaBCmax"_a, "etaA_BCmax"_a,
           "Structured U4U6 recoupling object.");
  m.def("u4u6db", &u4::u4u6db, py::return_value_policy::reference,
        "The global U(4) U-coefficient database (singleton).");

  // ---- U(4) 6-irrep recoupling: Z-coefficients (U4Z6) ----------------------
  py::class_<u4::U4Z6_db, std::unique_ptr<u4::U4Z6_db, py::nodelete>>(
      m, "U4Z6_db")
      .def("get_U4Z6",
           [](u4::U4Z6_db& db, const u4::irrep& u4B, const u4::irrep& u4A,
              const u4::irrep& u4ABC, const u4::irrep& u4C, const u4::irrep& u4AB,
              int etaABmax, int etaAB_Cmax, const u4::irrep& u4AC, int etaACmax,
              int etaAC_Bmax) {
             require_init();
             return db.get_U4Z6(u4B, u4A, u4ABC, u4C, u4AB, etaABmax, etaAB_Cmax,
                                u4AC, etaACmax, etaAC_Bmax);
           },
           "u4B"_a, "u4A"_a, "u4ABC"_a, "u4C"_a,
           "u4AB"_a, "etaABmax"_a, "etaAB_Cmax"_a,
           "u4AC"_a, "etaACmax"_a, "etaAC_Bmax"_a,
           "Structured U4Z6 recoupling object.");
  m.def("u4z6db", &u4::u4z6db, py::return_value_policy::reference,
        "The global U(4) Z-coefficient database (singleton).");

  // ---- U(4) 9-irrep recoupling coefficients (U4U9) -------------------------
  //  get_U4U9 returns a std::span view into a scratch buffer; we copy it into
  //  an owning U4U9 so the Python object stays valid across DB calls.
  py::class_<u4::U4U9_db, std::unique_ptr<u4::U4U9_db, py::nodelete>>(
      m, "U4U9_db")
      .def(
          "get_U4U9",
          [](u4::U4U9_db& db, const u4::irrep& u4A, const u4::irrep& u4B,
             const u4::irrep& u4C, const u4::irrep& u4D, const u4::irrep& u4AB,
             const u4::irrep& u4CD, const u4::irrep& u4AC, const u4::irrep& u4BD,
             const u4::irrep& u4ABCD, int etaACmax, int etaBDmax, int etaABCDmax,
             int etaABmax, int etaCDmax, int etaACBDmax) {
            require_init();
            auto src =
                db.get_U4U9(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD,
                            etaACmax, etaBDmax, etaABCDmax, etaABmax, etaCDmax,
                            etaACBDmax);
            u4::coeffs_t buf(src.data().begin(), src.data().end());
            return U4U9_t(std::move(buf), src.etaACmax(), src.etaBDmax(),
                          src.etaABCDmax(), src.etaABmax(), src.etaCDmax(),
                          src.etaACBDmax());
          },
          "u4A"_a, "u4B"_a, "u4C"_a, "u4D"_a, "u4AB"_a, "u4CD"_a, "u4AC"_a,
          "u4BD"_a, "u4ABCD"_a, "etaACmax"_a, "etaBDmax"_a, "etaABCDmax"_a,
          "etaABmax"_a, "etaCDmax"_a, "etaACBDmax"_a,
          "Structured U4U9 recoupling object (owning copy).");
  m.def("u4u9db", &u4::u4u9db, py::return_value_policy::reference,
        "The global U(4) 9-irrep coefficient database (singleton).");
}
