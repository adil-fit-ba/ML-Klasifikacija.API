using Microsoft.AspNetCore.Mvc;
using ML_Klasifikacija.Model;
using ML_Klasifikacija.Model.Helper;

namespace ML_Klasifikacija.API.Controllers
{
    [ApiController]
    public class HomeController : Controller
    {
        [HttpGet("Ulaz")]
        public IActionResult Ulaz()
        {
            MojDataSet fullDataSet = ExcelAlati.Ucitaj("Files/podaci3.xlsx", "OutletSize");

            (MojDataSet treningSet, MojDataSet testSet) = fullDataSet.Podijeli(testProcenat: 0.2, random_state: 42);

            StabloKlasifikator treningSetStabloKlasifikator = new StabloKlasifikator(treningSet);

            var x = fullDataSet.Evaluiraj(treningSetStabloKlasifikator, testSet);

            return Ok(new { x });
        }
    }
}
