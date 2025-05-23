using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Klasifikacija.Model.Helper
{
    public enum TipAtributa
    {
        Kategoricki,
        Numericki,
    }
    public class AtributMeta
    {
        public string Naziv { get; set; }
        public TipAtributa TipAtributa { get; set; }
    }
    public class RedPodatka
    {
        /// <summary>
        /// jedan red iz data seta, npr  Outlook: "sunny"
        /// </summary>
        ///
        public Dictionary<string, string> Atributi { get; set; }
        /// <summary>
        ///  Oznaka ciljne varijable
        /// </summary>
        /// <example>
        ///     "Play" ili  "Don't Play"
        /// </example>
        public string Klasa { get; set; }

    }
    public class MojDataSet
    {
        public string CiljnaVarijabla { get; set; }
        public List<RedPodatka> Podaci { get; set; }
        public List<AtributMeta> Atributi { get; set; }

        public MojDataSet(List<RedPodatka> podaci, string ciljnaVarijabla)
        {
            if (podaci == null || podaci.Count == 0)
            {
                throw new ArgumentException("Podaci ne mogu biti null ili prazni.");
            }

            Podaci = podaci;
            CiljnaVarijabla = ciljnaVarijabla;

            Atributi = podaci[0].Atributi
                .Select(kvp => new AtributMeta
                {
                    Naziv = kvp.Key,
                    TipAtributa = double.TryParse(kvp.Value, out _)  //todo: provjeriti čitav dataaset, ne samo prvi red, provjeriti da nije prazan string
                        ? TipAtributa.Numericki
                        : TipAtributa.Kategoricki
                }).ToList();
            
            
            var kategoricki =  Atributi.Where(x => x.TipAtributa == TipAtributa.Kategoricki)
                .Select(x=>x.Naziv)
                .ToList();


            var numericki = Atributi.Where(x => x.TipAtributa == TipAtributa.Numericki)
                .Select(x => x.Naziv)
                .ToList();

            Console.WriteLine(string.Join(" | ", kategoricki));
        }

        public (MojDataSet train, MojDataSet test) Podijeli(double testProcenat = 0.2, int? random_state = null)
        {
            //todo: implementirati podjelu sa random_state, da bude konzistentno
            var random = new Random(random_state ?? DateTime.Now.Millisecond);

            List<RedPodatka> randomRedoslijed = Podaci.OrderBy(x=> random.Next()).ToList();
            int testVelicina = (int)(randomRedoslijed.Count * testProcenat);

            List<RedPodatka> testPodaci = randomRedoslijed.Take(testVelicina).ToList();
            List<RedPodatka> trainPodaci = randomRedoslijed.Skip(testVelicina).ToList();

            var train = new MojDataSet(trainPodaci, CiljnaVarijabla);
            var test = new MojDataSet(testPodaci, CiljnaVarijabla);

            return (train, test);
        }

        public EvaluacijaRezultat Evaluiraj(StabloKlasifikator treningSetStabloKlasifikator, MojDataSet testSkup)
        {
            int tacni = 0;

            foreach (var red in testSkup.Podaci)
            {
                var predikcija = treningSetStabloKlasifikator.Predikcija(red);
                if (predikcija == red.Klasa)
                {
                    tacni++;
                }
            }

            return new EvaluacijaRezultat
            {
                Accuracy = tacni / (double)testSkup.Podaci.Count,
                Precision = 0,
                Recall = 0,
                F1Score = 0,
            };
        }
    }
}
