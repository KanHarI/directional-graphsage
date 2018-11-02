
import parsec as P


@P.generate
def num_i():
	tmp = yield P.many1(P.digit())
	return int(''.join(tmp))

@P.generate
def white():
	yield P.one_of(' \t\r\n')

@P.generate
def transparent():
	yield P.many(white)

@P.generate
def seperator():
	yield white >> transparent

class SdfHeader:
	def __init__(self, mol_num, atom_num, bond_num):
		self.mol_num = mol_num
		self.atom_num = atom_num
		self.bond_num = bond_num

@P.generate
def sdf_header():
	mol_num = yield transparent >> num_i
	atom_num = yield seperator >> num_i
	bond_num = yield seperator >> num_i
	yield P.times(seperator >> num_i, 5)
	yield seperator >> P.string("V2000")
	return SdfHeader(mol_num, atom_num, bond_num)

@P.generate
def num_f():
	tmp_1 = yield P.many(P.digit())
	tmp_2 = yield P.string('.') >> P.many(P.digit())
	return float(''.join(['0']+tmp_1+['.']+tmp_2+['0']))

class SdfAtom:
	def __init__(self, symb, dd, ccc):
		self.symb = symb
		self.dd = dd
		self.ccc = ccc

@P.generate
def sdf_atom():
	yield P.times(seperator >> num_f, 3) # coordinates
	atom_symb = yield seperator >> P.many(P.letter())
	atom_symb = ''.join(atom_symb)
	dd = yield seperator >> num_i # mass delta
	ccc = yield seperator >> num_i # charge delta
	yield P.times(seperator >> num_i, 4)
	return SdfAtom(atom_symb, dd, ccc)

class SdfBond:
	def __init__(self, fst, snd, bond_type):
		self.fst = fst
		self.snd = snd
		self.bond_type = bond_type

@P.generate
def sdf_bond():
	fst = yield seperator >> num_i
	snd = yield seperator >> num_i
	bond_type = yield seperator >> num_i
	yield P.times(seperator >> num_i, 3)
	return SdfBond(fst, snd, bond_type)

class SdfMolecule:
	def __init__(self, header, atoms, bonds, value):
		self.header = header
		self.atoms = atoms
		self.bonds = bonds
		self.value = value

	def get_value(self):
		return [1,0] if self.value>0 else [0,1]

@P.generate
def sdf_molecule():
	header = yield sdf_header
	atoms = yield P.times(sdf_atom, header.atom_num)
	bonds = yield P.times(sdf_bond, header.bond_num)
	yield seperator >> P.string('M') >> seperator >> P.string('END') >> seperator >> P.string('> <value>')
	value = yield seperator >> num_f
	yield seperator >> P.string('$$$$')
	return SdfMolecule(header, atoms, bonds, value)

class SdfFile:
	def __init__(self, file_name):
		c = open(file_name, 'rb').read().decode('utf-8')
		self.molecules = P.many1(sdf_molecule).parse(c)
