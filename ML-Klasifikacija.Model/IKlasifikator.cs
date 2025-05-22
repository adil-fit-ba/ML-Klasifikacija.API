using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Klasifikacija.Model
{
    public interface IKlasifikator
    {
        string Predikcija(RedPodatka redPodatka);
    }
}
