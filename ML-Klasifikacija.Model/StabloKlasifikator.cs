using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Klasifikacija.Model
{

    public class CvorStabla
    {
        public string? Atribut { get; set; } = null; // naziv atributa koji se koristi za deljenje
        public Dictionary<string, CvorStabla> Djeca { get; set; } = new();
        public string? Klasa { get; set; } = null; // koristimo samo za čvorove koji su listovi
        public bool JeList => Klasa != null; // da li je čvor list
    }


    public class StabloKlasifikator
    {
        public CvorStabla _korjen = null;
        public StabloKlasifikator(MojDataSet podaci)
        {
            if (!podaci.Atributi.Any())
            {
                 throw new ArgumentException("Podaci ne mogu biti null ili prazni.");
            }
            _korjen = IzgradiStablo(podaci);
        }

        public CvorStabla IzgradiStablo(MojDataSet podaci)
        {
            if (podaci.Podaci.Select(x=> x.Klasa).Distinct().Count() == 1)
            {
                // svi podaci imaju istu klasu
                return new CvorStabla { Klasa = podaci.Podaci.First().Klasa }; //ovo je list
            }
            // ovo nije list
            // rekurzivno delimo podatke


        }
    }

}
