using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DecisionTree.Model.Model.MLP.MLPMreza;

public class Layer
{
    public List<Neuron> Neuroni { get; set; } = new();

    public Layer(int brojNeurona, int brojUlazaPoNeuronu, Func<double, double> aktivacijskaFunkcija, Func<double, double> derivacija)
    {
        for (int i = 0; i < brojNeurona; i++)
            Neuroni.Add(new Neuron(brojUlazaPoNeuronu, aktivacijskaFunkcija, derivacija));
    }

    public double[] Izracunaj(double[] ulazi)
    {
        return Neuroni.Select(n => n.Izracunaj(ulazi)).ToArray();
    }
}