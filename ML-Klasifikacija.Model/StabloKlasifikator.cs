using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ML_Klasifikacija.Model.Helper;

namespace ML_Klasifikacija.Model
{

    public class CvorStabla
    {
        public string? Atribut { get; set; } = null; // naziv atributa koji se koristi za deljenje, koristimo samo ako nije list
        public Dictionary<string, CvorStabla> Djeca { get; set; } = new();
        public string? Klasa { get; set; } = null; // koristimo samo za čvorove koji su listovi
        public bool JeList => Klasa != null; // da li je čvor list
    }


    public class StabloKlasifikator: IKlasifikator
    {
        public CvorStabla _korjen = null;
        public StabloKlasifikator(MojDataSet dataSet)
        {
            if (!dataSet.Atributi.Any())
            {
                 throw new ArgumentException("Podaci ne mogu biti null ili prazni.");
            }
            _korjen = IzgradiStabloRekurzija(dataSet.Podaci, dataSet.Atributi);
        }

        private CvorStabla IzgradiStabloRekurzija(List<RedPodatka> Podaci, List<AtributMeta> Atributi)
        {
            if (Podaci.Select(x=> x.Klasa).Distinct().Count() == 1)
            {
                // svi podaci imaju istu klasu
                return new CvorStabla { Klasa = Podaci.First().Klasa }; //ovo je list
            }

            if (!Atributi.Any())
            {
                return new CvorStabla
                {
                    Klasa = NajcescaKlasa(podaci: Podaci),
                };
            }

            // ovo nije list
            // rekurzivno delimo podatke

            var kandidati = Atributi.Where(x => x.TipAtributa == TipAtributa.Kategoricki).ToList();

            var najbolji = kandidati.Select(a=>new {Atribut = a.Naziv, Gini = IzracunajGiniIndeks(Podaci, a.Naziv) })
                .OrderBy(x => x.Gini)
                .Select(x=>x.Atribut)
                .First();

            CvorStabla podstablo = new CvorStabla { Atribut = najbolji };

            var vrijednosti = Podaci.Select(p => p.Atributi[najbolji]).Distinct().ToArray(); //sunny overcast rain

            foreach (var vr in vrijednosti)
            {
                var podskup = Podaci.Where(x => x.Atributi[najbolji] == vr).ToList();
                var preostali = kandidati.Where(x => x.Naziv != najbolji).ToList();

                podstablo.Djeca[vr] = podskup.Any() ? IzgradiStabloRekurzija(podskup, preostali) :
                    new CvorStabla { Klasa = NajcescaKlasa(Podaci) };
            }

            /*
                todo: maksimalnu dubinu stabla
                                
             */
            return podstablo;
        }

        private string NajcescaKlasa(List<RedPodatka> podaci) => podaci.GroupBy(x => x.Klasa).OrderByDescending(x => x.Count()).First().Key;

        private double IzracunajGiniIndeks(List<RedPodatka> podaci, string atribut)
        {
            // https://hrcak.srce.hr/file/151776
            // pogledati primjer u excelu play1.xlsx ->
            // gini index outlook = 0.46
            // gini index windy = 0.23

            var grupe = podaci.GroupBy(p => p.Atributi[atribut]);
            double ukupno = podaci.Count;
            double gini = 0.0;

            foreach (var grupa in grupe)
            {
                double velicina = grupa.Count();
                double skor = grupa.GroupBy(p => p.Klasa)
                                    .Select(g => Math.Pow(g.Count() / velicina, 2))
                                    .Sum();
                gini += (1 - skor) * (velicina / ukupno);
            }

            return gini;
        }

        public string Predikcija(RedPodatka redPodatka)
        {
           return PredikcijaRekurzija(_korjen, redPodatka);
        }

        private string PredikcijaRekurzija(CvorStabla cvor, RedPodatka redPodatka)
        {
            if (cvor.JeList)
            {
                return cvor.Klasa;
            }

            if (!redPodatka.Atributi.ContainsKey(cvor.Atribut))
            {
                return "nepoznato";
            }

            string vrijednost = redPodatka.Atributi[cvor.Atribut];
            
            if (cvor.Djeca.TryGetValue(vrijednost, out var dijete))
            {
                return PredikcijaRekurzija(cvor.Djeca[vrijednost], redPodatka);
            }
            else
            {
                return "nepoznato";
            }
        }
    }

}
