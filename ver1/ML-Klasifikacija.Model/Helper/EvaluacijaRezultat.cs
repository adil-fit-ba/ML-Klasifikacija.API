using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Klasifikacija.Model.Helper
{
    public class EvaluacijaRezultat
    {
        public double Accuracy { get; internal set; }
        public int Precision { get; internal set; }
        public int Recall { get; internal set; }
        public int F1Score { get; internal set; }
    }
}
