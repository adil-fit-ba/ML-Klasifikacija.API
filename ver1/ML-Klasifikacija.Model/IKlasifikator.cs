using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ML_Klasifikacija.Model.Helper;

namespace ML_Klasifikacija.Model
{
    public interface IKlasifikator
    {
        string Predikcija(RedPodatka redPodatka);
    }
}
