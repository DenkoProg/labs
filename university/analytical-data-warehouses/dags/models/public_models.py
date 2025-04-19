from models.base_essentials import PublicBase
from sqlalchemy import Column, Integer, String, Boolean, Date, Time, Numeric, DateTime, Text, BigInteger, SmallInteger


class TestTable(PublicBase):
    __tablename__ = 'test_table'

    id = Column(BigInteger, primary_key=True)
    source_payment_charge_id = Column(Text)
    is_restored = Column(Boolean)


class TestTable2(PublicBase):
    __tablename__ = 'test_table2'

    id = Column(BigInteger, primary_key=True)
    source_payment_charge_id = Column(Text)
    is_restored = Column(Boolean)
